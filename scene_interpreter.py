import threading
import time
import json
import logging
import requests
from typing import Dict, List, Optional, Any
from dataclasses import asdict


class SceneInterpreter:
    """
    Interprets the scene using an LLM via OpenRouter API.
    Handles both passive "thinking" and active chat interactions.
    """

    def __init__(self, api_key: str, model: str, system_prompt: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.logger = logging.getLogger(__name__)

        if not system_prompt:
            self.system_prompt = (
                "You are an AI assistant analyzing a video feed. "
                "You will be provided with a list of detected objects, their positions (zones), "
                "and confidence levels. "
                "Your goal is to interpret the scene, describe what is happening, "
                "and answer user questions based on this data. "
                "Keep your responses concise and relevant to the visual context."
            )
        else:
            self.system_prompt = system_prompt

        self.latest_response = ""
        self.is_processing = False
        self.last_request_time = 0
        self.conversation_history = []

        # Threading lock for thread-safe access to latest_response
        self.lock = threading.Lock()

    def _format_scene_data(self, tracking_data: Dict[str, Any]) -> str:
        """
        Formats the tracking data into a descriptive string for the LLM.
        """
        if not tracking_data:
            return "The scene is currently empty. No objects detected."

        description = "Current Scene Objects:\n"

        # Group objects by class for better readability
        objects_by_class = {}
        for obj_id, obj in tracking_data.items():
            # Handle both dict and object (if dataclass)
            if not isinstance(obj, dict):
                obj = asdict(obj)

            class_name = obj.get("class_name", "unknown")
            if class_name not in objects_by_class:
                objects_by_class[class_name] = []
            objects_by_class[class_name].append(obj)

        for class_name, objects in objects_by_class.items():
            description += f"- {len(objects)} {class_name}(s):\n"
            for obj in objects:
                zone = obj.get("zone", "unknown location")
                confidence = obj.get("confidence", 0.0)
                # time_ago = time.time() - obj.get('last_seen', time.time())
                description += f"  * Located in {zone} (confidence: {confidence:.2f})\n"

        return description

    def interpret_scene(
        self,
        tracking_data: Dict[str, Any],
        user_prompt: Optional[str] = None,
        mode: str = "passive",
    ):
        """
        Triggers an interpretation of the scene.

        Args:
            tracking_data: Dictionary of tracked objects.
            user_prompt: Optional specific question from the user.
            mode: "passive" for periodic thinking, "chat" for user interaction.
        """
        if self.is_processing:
            self.logger.info("Skipping interpretation request: AI is busy.")
            return

        scene_description = self._format_scene_data(tracking_data)

        if mode == "passive":
            prompt = f"{scene_description}\n\nBased on these objects, briefly describe what is likely happening in the scene. Keep it under 2 sentences."
            # For passive mode, we don't necessarily need history, or maybe just a sliding window.
            # Let's keep it simple: single shot for passive.
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            # Chat mode
            prompt = f"{scene_description}\n\nUser Question: {user_prompt}"

            # Append to history
            # We construct a temporary message list including history + current context
            # Note: We might want to limit history size to avoid token limits
            messages = [{"role": "system", "content": self.system_prompt}]
            messages.extend(self.conversation_history[-10:])  # Keep last 10 turns
            messages.append({"role": "user", "content": prompt})

        # Start processing in a separate thread
        thread = threading.Thread(
            target=self._call_openrouter_api, args=(messages, mode)
        )
        thread.daemon = True
        thread.start()

    def _call_openrouter_api(self, messages: List[Dict[str, str]], mode: str):
        """
        Makes the actual API call to OpenRouter.
        """
        self.is_processing = True
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/JayNightmare/Object-Detect",  # Required by OpenRouter
                "X-Title": "Object Detect Camera App",
            }

            data = {"model": self.model, "messages": messages}

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10,  # Timeout to prevent hanging
            )

            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]

                with self.lock:
                    self.latest_response = content

                if mode == "chat":
                    self.conversation_history.append(messages[-1])  # User prompt
                    self.conversation_history.append(
                        {"role": "assistant", "content": content}
                    )

                self.logger.info(f"AI Response ({mode}): {content}")
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                with self.lock:
                    self.latest_response = f"AI Error: {response.status_code}"

        except Exception as e:
            self.logger.error(f"Exception during API call: {e}")
            with self.lock:
                self.latest_response = "AI Error: Connection failed"
        finally:
            self.is_processing = False
            self.last_request_time = time.time()

    def get_latest_response(self) -> str:
        """
        Returns the most recent AI response.
        """
        with self.lock:
            return self.latest_response
