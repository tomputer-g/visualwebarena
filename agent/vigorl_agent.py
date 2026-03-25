"""ViGORL agent for VisualWebArena.

Runs a Qwen2.5-VL ViGORL checkpoint locally in a multi-turn loop.
Non-tool-call turns follow the existing VWA SOM prompt template
(p_som_cot_id_actree_3s.json or any MultimodalCoT-compatible JSON).
Tool-call turns execute the ViGORL zoom-crop mechanism and feed the
crop back as an <observation> message.  Coordinate outputs in
<answer>…</answer> are mapped to the nearest SOM element ID via
the bounding-box data stored in the VWA observation metadata.


python run.py \
  --agent_type vigorl \
  --provider vigorl \
  --model gsarch/ViGoRL-Multiturn-3b-Web-Grounding \
  --instruction_path agent/prompts/jsons/p_som_cot_id_actree_3s.json \
  --observation_type image_som \
  --action_set_tag som \
  --temperature 0.5 \
  --max_tokens 512 \
  --max_steps 10 \
  --result_dir results/vigorl \
  [... standard VWA site/task args]

"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from browser_env import Trajectory
from browser_env.actions import (
    Action,
    ActionParsingError,
    create_id_based_action,
    create_none_action,
)
from browser_env.env_config import URL_MAPPINGS
from browser_env.utils import StateInfo
from llms import lm_config as lm_config_module
from llms.providers.vigorl_utils import (
    TOOL_RE,
    get_point_crop,
    parse_tool_coordinate,
)

# Import Agent base class. Use a direct module import to avoid a circular
# dependency through agent/__init__.py (which imports vigorl_agent itself).
from agent.agent import Agent

# ---------------------------------------------------------------------------
# Debug flag — set to False to silence all [ViGORL] diagnostic prints.
# ---------------------------------------------------------------------------
VIGORL_DEBUG: bool = True


def _dbg(*args: Any, **kwargs: Any) -> None:
    """Print only when VIGORL_DEBUG is True."""
    if VIGORL_DEBUG:
        print("[ViGORL]", *args, flush=True, **kwargs)


class ViGORLAgent(Agent):
    """Prompt agent that wraps a locally loaded ViGORL checkpoint.

    The SOM prompt JSON supplies the system prompt, few-shot examples, and
    observation template.  ViGORL's internal multi-turn zoom loop is handled
    transparently; the outer VWA harness sees a single Action per step.
    """

    def __init__(
        self,
        model_name: str,
        instruction_path: str,
        action_set_tag: str,
        lm_config: lm_config_module.LMConfig,
    ) -> None:
        super().__init__()
        with open(instruction_path) as f:
            self.instruction = json.load(f)
        # Normalise examples to tuples (obs_text, action_text, img_path).
        self.instruction["examples"] = [
            tuple(e) for e in self.instruction["examples"]
        ]
        self.model_name = model_name
        self.action_set_tag = action_set_tag
        self.lm_config = lm_config
        self.obs_modality: str = self.instruction["meta_data"]["observation"]
        # Lazy-loaded; heavy GPU allocation deferred until first call.
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        _dbg(f"Loading {self.model_name} …")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            max_pixels=12960000,
            min_pixels=3136,
        )

    # ------------------------------------------------------------------
    # URL helpers (mirror PromptConstructor)
    # ------------------------------------------------------------------

    def _map_url_to_real(self, url: str) -> str:
        original = url
        for local, real in URL_MAPPINGS.items():
            if local in url:
                url = url.replace(local, real)
        if url != original:
            _dbg(f"URL → real:  {original!r}  →  {url!r}")
        return url

    def _map_url_to_local(self, url: str) -> str:
        original = url
        for local, real in URL_MAPPINGS.items():
            if real in url:
                url = url.replace(real, local)
            if real.replace("http", "https") in url:
                url = url.replace(real.replace("http", "https"), local)
        if url != original:
            _dbg(f"URL → local: {original!r}  →  {url!r}")
        return url

    # ------------------------------------------------------------------
    # Message construction (Qwen chat format)
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        page_screenshot_img: Image.Image,
        images: list[Image.Image],
    ) -> list[dict]:
        """Build the initial Qwen-format message list from the VWA SOM template."""
        intro = self.instruction["intro"]
        examples = self.instruction["examples"]
        template = self.instruction["template"]

        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        obs: str = state_info["observation"][self.obs_modality]

        # Truncate observation text to stay within context budget.
        max_obs_length = self.lm_config.gen_config.get("max_obs_length")
        if max_obs_length and len(obs) > max_obs_length:
            _dbg(f"Truncating obs: {len(obs)} → {max_obs_length} chars")
            obs = obs[:max_obs_length]

        url = state_info["info"]["page"].url
        previous_action_str = meta_data["action_history"][-1]
        mapped_url = self._map_url_to_real(url)
        _dbg(f"Building messages | intent: {intent!r}")
        _dbg(f"  page URL: {url!r}")
        _dbg(f"  prev action: {previous_action_str!r}")
        _dbg(f"  obs length: {len(obs)} chars | {len(self.instruction['examples'])} few-shot examples")
        current = template.format(
            objective=intent,
            url=mapped_url,
            observation=obs,
            previous_action=previous_action_str,
        )

        messages: list[dict] = [{"role": "system", "content": intro}]

        # Few-shot examples (image + obs text  →  action text).
        for obs_text, action_text, img_path in examples:
            example_img = Image.open(img_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text",  "text": obs_text},
                    {"type": "text",  "text": "IMAGES: (1) current page screenshot"},
                    {"type": "image", "image": example_img},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": action_text}],
            })

        # Current turn: formatted observation + SOM-annotated screenshot.
        user_content: list[dict] = [
            {"type": "text",  "text": current},
            {"type": "text",  "text": "IMAGES: (1) current page screenshot"},
            {"type": "image", "image": page_screenshot_img},
        ]
        for i, img in enumerate(images):
            user_content.append({"type": "text",  "text": f"({i + 2}) input image {i + 1}"})
            user_content.append({"type": "image", "image": img})

        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # Single generation step
    # ------------------------------------------------------------------

    def _generate(
        self, messages: list[dict], temperature: float, max_new_tokens: int
    ) -> str:
        from qwen_vl_utils import process_vision_info

        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text_prompt],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        do_sample = temperature > 0.0
        gen_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
        )
        trimmed = gen_ids[:, inputs.input_ids.shape[1]:]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _extract_action(self, response: str) -> Optional[str]:
        """Extract the action string from <answer>...</answer> tags."""
        m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL)
        if m:
            raw = m.group(1).strip()
            localised = self._map_url_to_local(raw)
            _dbg(f"Action extracted: {localised!r}")
            return localised
        _dbg("No <answer> tags found in response")
        return None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def next_action(
        self,
        trajectory: Trajectory,
        intent: str,
        meta_data: dict[str, Any],
        images: Optional[list[Image.Image]] = None,
        output_response: bool = False,
    ) -> Action:
        self._load_model()

        gc = self.lm_config.gen_config
        temperature    = gc.get("temperature", 0.5)
        max_new_tokens = gc.get("max_new_tokens", 512)
        max_turns      = gc.get("max_turns", 5)
        crop_offset    = gc.get("crop_offset", 182)
        crop_size      = gc.get("crop_size", 672)
        max_retry      = gc.get("max_retry", 3)

        state_info: StateInfo = trajectory[-1]  # type: ignore[assignment]
        page_screenshot_img = Image.fromarray(state_info["observation"]["image"])

        messages = self._build_messages(
            trajectory, intent, meta_data, page_screenshot_img, images or []
        )
        # Keep a reference to the original screenshot for all zoom crops.
        init_image = page_screenshot_img
        last_response = ""

        # ------------------------------------------------------------------
        # Multi-turn loop
        # ------------------------------------------------------------------
        for turn in range(1, max_turns + 1):
            _dbg(f"--- turn {turn}/{max_turns} ---")
            response = self._generate(messages, temperature, max_new_tokens)
            last_response = response
            _dbg(f"Model response:\n{response}")
            if output_response:
                print(f"[ViGORL turn {turn}]: {response}", flush=True)

            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            })

            # 1. Tool call → produce zoom crop and continue.
            # (Commented out: current prompt template uses single-turn <think> reasoning.)
            # tool_match = TOOL_RE.search(response)
            # if tool_match:
            #     coord = parse_tool_coordinate(tool_match.group(1))
            #     _dbg(f"Tool call detected | raw: {tool_match.group(1)!r} | parsed coord: {coord}")
            #     if coord is not None:
            #         crop = get_point_crop(
            #             init_image, coord,
            #             offset=crop_offset, crop_size=crop_size,
            #         )
            #         _dbg(f"Zoom crop produced: offset={crop_offset} size={crop_size} → crop {crop.size}")
            #         messages.append({
            #             "role": "user",
            #             "content": [
            #                 {
            #                     "type": "text",
            #                     "text": (
            #                         "<observation>\n"
            #                         "Here is the crop of the image centered on the coordinate:\n"
            #                         "</observation>"
            #                     ),
            #                 },
            #                 {"type": "image", "image": crop},
            #             ],
            #         })
            #         continue  # Next turn with crop in context.
            #     else:
            #         _dbg("Tool call found but coordinate parsing failed; skipping zoom")

            # 2. Extract action from <answer>...</answer> tags.
            action_str = self._extract_action(response)
            if action_str is not None:
                for _ in range(max_retry):
                    try:
                        action = create_id_based_action(action_str)
                        action["raw_prediction"] = response
                        _dbg(f"Action parsed OK: {action_str!r}")
                        return action
                    except ActionParsingError as e:
                        _dbg(f"ActionParsingError on {action_str!r}: {e}")
                        break  # Malformed action string; fall through.
            else:
                _dbg("No <answer> tags found; continuing to next turn")

        # ------------------------------------------------------------------
        # Soft prompt: nudge the model toward a final <answer> action.
        # Mirrors the soft-prompt technique in demo_multiturn.py.
        # ------------------------------------------------------------------
        soft_prefix = "Based on all the information I've gathered, I'll now provide my final answer.\n<answer> "
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": soft_prefix}],
        })

        from qwen_vl_utils import process_vision_info

        soft_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        img_inputs, vid_inputs = process_vision_info(messages)
        soft_inputs = self._processor(
            text=[soft_text],
            images=img_inputs,
            videos=vid_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        _dbg("max_turns exhausted; issuing soft prompt for forced final answer")
        soft_ids = self._model.generate(
            **soft_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic for forced final answer
        )
        soft_trimmed = soft_ids[:, soft_inputs.input_ids.shape[1]:]
        last_response = soft_prefix + self._processor.batch_decode(
            soft_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        _dbg(f"Soft-prompt response:\n{last_response}")
        if output_response:
            print(f"[ViGORL soft-prompt]: {last_response}", flush=True)

        action_str = self._extract_action(last_response)
        if action_str is not None:
            try:
                action = create_id_based_action(action_str)
                action["raw_prediction"] = last_response
                _dbg(f"Soft-prompt action parsed OK: {action_str!r}")
                return action
            except ActionParsingError as e:
                _dbg(f"ActionParsingError on soft-prompt action {action_str!r}: {e}")

        _dbg("All parsing strategies failed; returning none action")
        action = create_none_action()
        action["raw_prediction"] = last_response
        return action

    def reset(self, test_config_file: str) -> None:  # noqa: D401
        pass
