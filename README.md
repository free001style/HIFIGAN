<h1 align="center">Neural Vocoder</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
   <a href="#final-results">Final results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository contains the implementation of [HiFiGAN](https://arxiv.org/pdf/2010.05646) vocoder.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw3_nv).

See [WandB report](https://wandb.ai/free001style/HiFiGAN/reports/HiFiGAN-vocoder--VmlldzoxMDQ1OTQ1NA) with implementation details and audio analysis.

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment
   using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n nv python=3.11

   # activate env
   conda activate nv
   ```

1. Install all required packages.

   ```bash
   pip install -r requirements.txt
   ```
2. Download model checkpoint.

   ```bash
   python download_weights.py
   ```

## How To Use

### Inference

1) If you only want to synthesize one text/phrase and save it, run the following command:

   ```bash
   python synthesize.py 'text="YOUR_TEXT"' save_path=SAVE_PATH
   ```
   where `SAVE_PATH` is a path to save synthesize audio. Please be careful in quotes.

2) If you want to synthesize audio from text files, your directory with text should has the following format:
   ```
   NameOfTheDirectoryWithUtterances
   └── transcriptions
        ├── UtteranceID1.txt
        ├── UtteranceID2.txt
        .
        .
        .
        └── UtteranceIDn.txt
   ```
   Run the following command:
   ```bash
   python synthesize.py dir_path=DIR_PATH save_path=SAVE_PATH
   ```
   where `DIR_PATH` is directory with text and `SAVE_PATH` is a path to save synthesize audio.

### Training

To reproduce this model, run the following command:

   ```bash
   python train.py
   ```

It takes around hours to train model from scratch on A100 GPU.

## Final results

- `Mihajlo Pupin was a founding member of National Advisory Committee for Aeronautics (NACA) on 3 March 1915, which later became NASA, and he participated in the founding of American Mathematical Society and American Physical Society.`

   <video src='https://github.com/user-attachments/assets/aad2700f-b51b-4b42-bf89-e28ff011fc71' width=180/>

- `Leonard Bernstein was an American conductor, composer, pianist, music educator, author, and humanitarian. Considered to be one of the most important conductors of his time, he was the first American-born conductor to receive international acclaim.`

   <video src='https://github.com/user-attachments/assets/ee730e23-279a-4128-860e-01164e27fec5' width=180/>

- `Lev Termen, better known as Leon Theremin was a Russian inventor, most famous for his invention of the theremin, one of the first electronic musical instruments and the first to be mass-produced.`

   <video src='https://github.com/user-attachments/assets/e01c68e8-c51f-47f6-b100-30703681d713' width=180/>

- `Deep Learning in Audio course at HSE University offers an exciting and challenging exploration of cutting-edge techniques in audio processing, from speech recognition to music analysis. With complex homeworks that push students to apply theory to real-world problems, it provides a hands-on, rigorous learning experience that is both demanding and rewarding.`

   <video src='https://github.com/user-attachments/assets/0d2dba62-e058-4afe-b26e-930c0e673762' width=180/>

- `Dmitri Shostakovich was a Soviet-era Russian composer and pianist who became internationally known after the premiere of his First Symphony in 1926 and thereafter was regarded as a major composer.`

   <video src='https://github.com/user-attachments/assets/4b4965b0-6f57-4b38-a5f3-a289de705fcb' width=180/>

`WV-MOS=3.47` using text and `WV-MOS=2.15` using MelSpecs.
## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
