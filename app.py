# Modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
import os

gpt_path = os.environ.get(
    "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)
sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True"))


import gradio as gr
import librosa
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path
from time import time as ttime

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.mel_processing import spectrogram_torch
from module.models import SynthesizerTrn
from my_utils import load_audio
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


# bert_model=bert_model.to(device)
def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


n_semantic = 1024
dict_s2 = torch.load(sovits_path, map_location="cpu")
hps = dict_s2["config"]


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu")
config = dict_s1["config"]
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model
)
if is_half == True:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
hz = 50
max_sec = config["data"]["max_sec"]
# t2s_model = Text2SemanticLightningModule.load_from_checkpoint(checkpoint_path=gpt_path, config=config, map_location="cpu")#########todo
t2s_model = Text2SemanticLightningModule(config, "ojbk", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half == True:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {"Chinese": "zh", "English": "en", "Japanese": "ja"}


def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language):
    if len(prompt_text) > 100 or len(text) > 100:
        return
    t0 = ttime()
    prompt_text = prompt_text.strip("\n")
    prompt_language, text = prompt_language, text.strip("\n")
    with torch.no_grad():
        wav16k, _ = librosa.load(ref_wav_path, sr=16000)  # 派蒙
        # length of wav16k in sec should be in 60s
        if len(wav16k) < 16000 * 60:
            return
        wav16k = wav16k[: int(hps.data.sampling_rate * max_sec)]
        wav16k = torch.from_numpy(wav16k)
        if is_half == True:
            wav16k = wav16k.half().to(device)
        else:
            wav16k = wav16k.to(device)
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
    t1 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language)
    phones1 = cleaned_text_to_sequence(phones1)
    texts = text.split("\n")
    audio_opt = []
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    for text in texts:
        phones2, word2ph2, norm_text2 = clean_text(text, text_language)
        phones2 = cleaned_text_to_sequence(phones2)
        if prompt_language == "zh":
            bert1 = get_bert_feature(norm_text1, word2ph1).to(device)
        else:
            bert1 = torch.zeros(
                (1024, len(phones1)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
        if text_language == "zh":
            bert2 = get_bert_feature(norm_text2, word2ph2).to(device)
        else:
            bert2 = torch.zeros((1024, len(phones2))).to(bert1)
        bert = torch.cat([bert1, bert2], 1)

        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=config["inference"]["top_k"],
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(
        np.int16
    )


initial_md = """
# GPT-SoVITS Zero-shot TTS Demo

https://github.com/RVC-Boss/GPT-SoVITS

*I'm not the author of this model, and I just borrowed it to make a demo.*

- *Input text is limited to 100 characters.*
- *Input audio is limited to 60 seconds.*

**License**

https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE

This software is open source under the MIT License, the author does not have any control over the software, and the user is solely responsible for the use of the software and for the distribution of the sound derived from the software. 
If you do not agree with these terms and conditions, you may not use or reference any of the code or files in the package. 
"""

with gr.Blocks(title="GPT-SoVITS Zero-shot TTS Demo") as app:
    gr.Markdown(initial_md)
    with gr.Group():
        gr.Markdown(value="*Upload reference audio")
        with gr.Row():
            inp_ref = gr.Audio(label="Reference audio", type="filepath")
            prompt_text = gr.Textbox(label="Transcription of reference audio")
            prompt_language = gr.Dropdown(
                label="Language of reference audio",
                choices=["Chinese", "English", "Japanese"],
                value="Japanese",
            )
        gr.Markdown(value="*Text to synthesize")
        with gr.Row():
            text = gr.Textbox(label="Text to synthesize")
            text_language = gr.Dropdown(
                label="Language of text",
                choices=["Chinese", "English", "Japanese"],
                value="Japanese",
            )
            inference_button = gr.Button("Synthesize", variant="primary")
            output = gr.Audio(label="Result")
        inference_button.click(
            get_tts_wav,
            [inp_ref, prompt_text, prompt_language, text, text_language],
            [output],
        )

app.launch(inbrowser=True)
