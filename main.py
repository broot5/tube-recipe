from typing import Literal
from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from openai import OpenAI
import yt_dlp  # type: ignore


class Settings(BaseSettings):
    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1-mini"
    video_id: str = ""
    preferred_languages: str = "ko"

    @property
    def preferred_languages_list(self) -> list[str]:
        return [lang.strip() for lang in self.preferred_languages.split(",")]

    class Config:
        env_file = ".env"


settings = Settings()


class VideoInfo(BaseModel):
    title: str
    author: str
    thumbnail_url: str


class HowToStep(BaseModel):
    type: Literal["HowToStep"] = Field(default="HowToStep", alias="@type")
    text: str


class HowToSection(BaseModel):
    type: Literal["HowToSection"] = Field(default="HowToSection", alias="@type")
    name: str
    itemListElement: list[HowToStep]


class BaseRecipe(BaseModel):
    context: Literal["https://schema.org"] = Field(
        default="https://schema.org", alias="@context"
    )
    type: Literal["Recipe"] = Field(default="Recipe", alias="@type")
    name: str
    cookTime: str
    description: str
    keywords: str
    prepTime: str
    recipeCategory: str
    recipeCuisine: str
    recipeIngredient: list[str]
    recipeInstructions: list[HowToStep | HowToSection]


class Recipe(BaseRecipe):
    image: str
    author: str

    @classmethod
    def from_base_recipe(cls, base_recipe: BaseRecipe, video_info: VideoInfo):
        return cls(
            **base_recipe.model_dump(),
            image=video_info.thumbnail_url,
            author=video_info.author,
        )


def get_video_info(video_id: str) -> VideoInfo | None:
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}",
                download=False,
            )

            if info is None:
                return None

            return VideoInfo(
                title=info.get("title") or "",
                author=info.get("uploader") or "",
                thumbnail_url=info.get("thumbnail") or "",
            )
    except Exception as e:
        print(f"Error fetching video info: {e}")
        return None


def get_transcript(video_id: str) -> str | None:
    try:
        transcript = YouTubeTranscriptApi().fetch(
            video_id, languages=settings.preferred_languages_list + ["en"]
        )
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None

    transcript_text = " ".join([entry["text"] for entry in transcript.to_raw_data()])
    return transcript_text


def extract_recipe(title: str, transcript: str) -> BaseRecipe | None:
    client = OpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
    )

    instruction = """
    You are a professional recipe extraction assistant. Extract a cooking recipe from the given YouTube video title and transcript, then convert it into structured JSON-LD format.

    Follow these specific guidelines:

    1. **name**: Extract the exact dish name from the title or transcript
    2. **description**: Write a 2-3 sentence description of the dish and its characteristics
    3. **keywords**: Create relevant keywords separated by commas (cuisine type, cooking method, difficulty, main ingredients)
    4. **recipeCategory**: Choose from: "appetizer", "main course", "side dish", "dessert", "beverage", "snack"
    5. **recipeCuisine**: Specify cuisine type (e.g., "Korean", "Italian", "Chinese", "American", "Japanese")
    6. **recipeIngredient**: List all ingredients with exact quantities and units as mentioned in the video
    7. **cookTime**: Estimate total cooking time in ISO 8601 duration format (e.g., "PT30M" for 30 minutes)
    8. **prepTime**: Estimate preparation time in ISO 8601 duration format
    9. **recipeInstructions**: Break down cooking steps comprehensively:
       - Use HowToStep for individual cooking actions, techniques, and specific instructions
       - Use HowToSection to group related steps (e.g., "Preparation", "Cooking", "Finishing")
       - Include ALL cooking tips, tricks, and helpful advice mentioned in the video
       - Include temperature settings, heat levels, timing cues, and visual indicators
       - Include any alternative methods or ingredient substitutions mentioned
       - Include storage tips, serving suggestions, and troubleshooting advice
       - Each step should be clear, actionable, and preserve the chef's original guidance

    **Important rules:**
    - Only extract recipe-related content, ignore unrelated video content
    - If quantities are vague, use reasonable estimates
    - Preserve original ingredient names and cooking terms
    - If cooking/prep times aren't mentioned, estimate based on the recipe complexity
    - For recipeInstructions, focus on the actual cooking process, not video commentary
    - Capture EVERY useful tip, technique, and piece of advice that would help someone cook this dish successfully
    - Include any warnings about common mistakes or things to avoid

    Output must be valid JSON-LD following schema.org Recipe format.    
    """

    try:
        response = client.responses.parse(
            model=settings.openai_model,
            instructions=instruction,
            input=f"""
            Title: {title}
            Transcript: {transcript}
            """,
            text_format=BaseRecipe,
        )
        return response.output_parsed
    except Exception as e:
        print(f"Error extracting recipe: {e}")
        return None


def main():
    video_id = settings.video_id

    transcript_text = get_transcript(video_id)
    if transcript_text is None:
        print("Failed to retrieve transcript.")
        return

    video_info = get_video_info(video_id)
    if video_info is None:
        print("Failed to retrieve video information.")
        return

    base_recipe = extract_recipe(title=video_info.title, transcript=transcript_text)

    if base_recipe is None:
        print("Failed to extract recipe")
        return

    recipe = Recipe.from_base_recipe(base_recipe, video_info)

    print(recipe.model_dump_json(by_alias=True))


if __name__ == "__main__":
    main()
