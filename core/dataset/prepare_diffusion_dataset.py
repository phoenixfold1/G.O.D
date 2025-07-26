import os
import shutil
import zipfile

from core import constants as cst


def prepare_dataset(
    training_images_zip_path: str,
    training_images_repeat: int,
    instance_prompt: str,
    class_prompt: str,
    job_id: str,
    regularization_images_dir: str = None,
    regularization_images_repeat: int = None,
):
    extraction_dir = f"{cst.DIFFUSION_DATASET_DIR}/tmp/{job_id}/"
    os.makedirs(extraction_dir, exist_ok=True)
    with zipfile.ZipFile(training_images_zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)

    extracted_items = [entry for entry in os.listdir(extraction_dir)]
    if len(extracted_items) == 1 and os.path.isdir(os.path.join(extraction_dir, extracted_items[0])):
        training_images_dir = os.path.join(extraction_dir, extracted_items[0])
    else:
        training_images_dir = extraction_dir

    IMAGE_STYLES = [
        "Watercolor Painting",
        "Oil Painting",
        "Digital Art",
        "Pencil Sketch",
        "Comic Book Style",
        "Cyberpunk",
        "Steampunk",
        "Impressionist",
        "Pop Art",
        "Minimalist",
        "Gothic",
        "Art Nouveau",
        "Pixel Art",
        "Anime",
        "3D Render",
        "Low Poly",
        "Photorealistic",
        "Vector Art",
        "Abstract Expressionism",
        "Realism",
        "Futurism",
        "Cubism",
        "Surrealism",
        "Baroque",
        "Renaissance",
        "Fantasy Illustration",
        "Sci-Fi Illustration",
        "Ukiyo-e",
        "Line Art",
        "Black and White Ink Drawing",
        "Graffiti Art",
        "Stencil Art",
        "Flat Design",
        "Isometric Art",
        "Retro 80s Style",
        "Vaporwave",
        "Dreamlike",
        "High Fantasy",
        "Dark Fantasy",
        "Medieval Art",
        "Art Deco",
        "Hyperrealism",
        "Sculpture Art",
        "Caricature",
        "Chibi",
        "Noir Style",
        "Lowbrow Art",
        "Psychedelic Art",
        "Vintage Poster",
        "Manga",
        "Holographic",
        "Kawaii",
        "Monochrome",
        "Geometric Art",
        "Photocollage",
        "Mixed Media",
        "Ink Wash Painting",
        "Charcoal Drawing",
        "Concept Art",
        "Digital Matte Painting",
        "Pointillism",
        "Expressionism",
        "Sumi-e",
        "Retro Futurism",
        "Pixelated Glitch Art",
        "Neon Glow",
        "Street Art",
        "Acrylic Painting",
        "Bauhaus",
        "Flat Cartoon Style",
        "Carved Relief Art",
        "Fantasy Realism",
    ]
    captions = []
    for caption_files in os.listdir(training_images_dir):
        if caption_files.endswith(".txt"):
            with open(os.path.join(training_images_dir, caption_files), "r") as f:
                caption = f.read()
            captions.append(caption)

    style_check = {style: 0 for style in IMAGE_STYLES}
    for caption in captions:
        for style in IMAGE_STYLES:
            if style in caption:
                style_check[style] += 1

    style_check = sorted(style_check.items(), key=lambda x: x[1], reverse=True)
    most_common_style_name = style_check[0][0]
    most_common_style_count = style_check[0][1]
    print(f"Most common style: {most_common_style_name} with {most_common_style_count} images")
    if most_common_style_count < len(captions):
        class_prompt = "style"
        instance_prompt = most_common_style_name
        training_images_repeat = 5
    else:
        class_prompt = "person"
        # find word that appears in most of the captions, that instance_prompt
        word_count = {word: 0 for word in captions[0].split()}
        for caption in captions:
            for word in caption.split():
                word_count[word] += 1
        most_common_word_name = max(word_count.items(), key=lambda x: x[1])[0]
        instance_prompt = most_common_word_name
        training_images_repeat = 10

    output_dir = f"{cst.DIFFUSION_DATASET_DIR}/{job_id}/"
    os.makedirs(output_dir, exist_ok=True)

    training_dir = os.path.join(
        output_dir,
        f"img/{training_images_repeat}_{instance_prompt} {class_prompt}",
    )

    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)

    shutil.copytree(training_images_dir, training_dir)

    if regularization_images_dir is not None:
        regularization_dir = os.path.join(
            output_dir,
            f"reg/{regularization_images_repeat}_{class_prompt}",
        )

        if os.path.exists(regularization_dir):
            shutil.rmtree(regularization_dir)
        shutil.copytree(regularization_images_dir, regularization_dir)

    if not os.path.exists(os.path.join(output_dir, "log")):
        os.makedirs(os.path.join(output_dir, "log"))

    if not os.path.exists(os.path.join(output_dir, "model")):
        os.makedirs(os.path.join(output_dir, "model"))

    if os.path.exists(extraction_dir):
        shutil.rmtree(extraction_dir)

    if os.path.exists(training_images_zip_path):
        os.remove(training_images_zip_path)
