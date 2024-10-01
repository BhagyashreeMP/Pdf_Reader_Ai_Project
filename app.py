import base64
from typing import cast
import pathlib
import gradio as gr
import spaces
import torch
from colpali_engine.models.paligemma.colpali import ColPali, ColPaliProcessor
from mistral_common.protocol.instruct.messages import (
    ImageURLChunk,
    TextChunk,
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_inference.generate import generate
from mistral_inference.transformer import Transformer
from pdf2image import convert_from_path
from torch.utils.data import DataLoader
from tqdm import tqdm

PIXTAL_MODEL_ID = "mistral-community--pixtral-12b-240910"
PIXTRAL_MODEL_SNAPSHOT = "95758896fcf4691ec9674f29ec90d1441d9d26d2"
PIXTRAL_MODEL_PATH = (
    pathlib.Path().home()
    / f".cache/huggingface/hub/models--{PIXTAL_MODEL_ID}/snapshots/{PIXTRAL_MODEL_SNAPSHOT}"
)


COLPALI_GEMMA_MODEL_ID = "vidore--colpaligemma-3b-pt-448-base"
COLPALI_GEMMA_MODEL_SNAPSHOT = "12c59eb7e23bc4c26876f7be7c17760d5d3a1ffa"
COLPALI_GEMMA_MODEL_PATH = (
    pathlib.Path().home()
    / f".cache/huggingface/hub/models--{COLPALI_GEMMA_MODEL_ID}/snapshots/{COLPALI_GEMMA_MODEL_SNAPSHOT}"
)
COLPALI_MODEL_ID = "vidore--colpali-v1.2"
COLPALI_MODEL_SNAPSHOT = "2d54d5d3684a4f5ceeefbef95df0c94159fd6a45"
COLPALI_MODEL_PATH = (
    pathlib.Path().home()
    / f".cache/huggingface/hub/models--{COLPALI_MODEL_ID}/snapshots/{COLPALI_MODEL_SNAPSHOT}"
)


def image_to_base64(image_path):
    with open(image_path, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded_string}"


@spaces.GPU(duration=60)
def pixtral_inference(
    images,
    text,
):
    if len(images) == 0:
        raise gr.Error("No images for generation")
    if text == "":
        raise gr.Error("No query for generation")
    tokenizer = MistralTokenizer.from_file(f"{PIXTRAL_MODEL_PATH}/tekken.json")
    model = Transformer.from_folder(PIXTRAL_MODEL_PATH)

    messages = [
        UserMessage(
            content=[ImageURLChunk(image_url=image_to_base64(i[0])) for i in images]
            + [TextChunk(text=text)]
        )
    ]

    completion_request = ChatCompletionRequest(messages=messages)

    encoded = tokenizer.encode_chat_completion(completion_request)

    images = encoded.images
    tokens = encoded.tokens

    out_tokens, _ = generate(
        [tokens],
        model,
        images=[images],
        max_tokens=512,
        temperature=0.45,
        eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id,
    )
    result = tokenizer.decode(out_tokens[0])
    return result


@spaces.GPU(duration=60)
def retrieve(query: str, ds, images, k):
    if len(images) == 0:
        raise gr.Error("No docs/images for retrieval")
    if query == "":
        raise gr.Error("No query for retrieval")

    model = ColPali.from_pretrained(
        COLPALI_GEMMA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    model.load_adapter(COLPALI_MODEL_PATH)
    model = model.eval()
    processor = cast(
        ColPaliProcessor, ColPaliProcessor.from_pretrained(COLPALI_MODEL_PATH)
    )

    qs = []
    with torch.no_grad():
        batch_query = processor.process_queries([query])
        batch_query = {k: v.to("cuda") for k, v in batch_query.items()}
        embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    scores = processor.score(qs, ds).numpy()
    top_k_indices = scores.argsort(axis=1)[0][-k:][::-1]
    results = []
    for idx in top_k_indices:
        results.append((images[idx], f"Score {scores[0][idx]:.2f}"))
    del model
    del processor
    torch.cuda.empty_cache()
    return results


def index(files, ds):
    images = convert_files(files)
    return index_gpu(images, ds)


def convert_files(files):
    images = []
    for f in files:
        images.extend(convert_from_path(f, thread_count=4))

    if len(images) >= 150:
        raise gr.Error("The number of images in the dataset should be less than 150.")
    return images


@spaces.GPU(duration=60)
def index_gpu(images, ds):
    model = ColPali.from_pretrained(
        COLPALI_GEMMA_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()

    model.load_adapter(COLPALI_MODEL_PATH)
    model = model.eval()
    processor = cast(
        ColPaliProcessor, ColPaliProcessor.from_pretrained(COLPALI_MODEL_PATH)
    )

    # run inference - docs
    dataloader = DataLoader(
        images,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: processor.process_images(x),
    )

    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to("cuda") for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
        ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
    del model
    del processor
    torch.cuda.empty_cache()
    return f"Uploaded and converted {len(images)} pages", ds, images


def get_example():
    return [
        [["plants_and_people.pdf"], "What is the global population in 2050 ? "],
        [["plants_and_people.pdf"], "Where was Teosinte domesticated ?"],
    ]


css = """
#title-container {
    margin: 0 auto;
    max-width: 800px;
    text-align: center;
}
#col-container {
    margin: 0 auto;
    max-width: 600px;
}
"""
file = gr.File(file_types=["pdf"], file_count="multiple", label="PDFs")
query = gr.Textbox("", placeholder="Enter your query here", label="Query")

with gr.Blocks(
    title="Document Question Answering with ColPali & Pixtral",
    theme=gr.themes.Soft(),
    css=css,
) as demo:
    with gr.Row(elem_id="title-container"):
        gr.Markdown("""# Document Question Answering with ColPali & Pixtral""")
    with gr.Column(elem_id="col-container"):
        with gr.Row():
            gr.Examples(
                examples=get_example(),
                inputs=[file, query],
            )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("## Index PDFs")
                file.render()
                convert_button = gr.Button("🔄 Run", variant="primary")
                message = gr.Textbox("Files not yet uploaded", label="Status")
                embeds = gr.State(value=[])
                imgs = gr.State(value=[])
                img_chunk = gr.State(value=[])

            with gr.Column(scale=3):
                gr.Markdown("## Retrieve with ColPali and answer with Pixtral")
                query.render()
                k = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    label="Number of docs to retrieve",
                    value=1,
                )
                answer_button = gr.Button("🏃 Run", variant="primary")

        output_gallery = gr.Gallery(
            label="Retrieved docs", height=400, show_label=True, interactive=False
        )
        output = gr.Textbox(label="Answer", lines=2, interactive=False)

        convert_button.click(
            index, inputs=[file, embeds], outputs=[message, embeds, imgs]
        )
        answer_button.click(
            retrieve, inputs=[query, embeds, imgs, k], outputs=[output_gallery]
        ).then(pixtral_inference, inputs=[output_gallery, query], outputs=[output])


if __name__ == "__main__":
    demo.queue(max_size=10).launch()
