<img src="./image.png" alt="GemOCR" width="25%">

#      GemOCR

GemOCR is your ultimate toolkit for reviving old RPG sourcebooks, blending the power of Optical Character Recognition (OCR) with cutting-edge AI. Standard OCR tools struggle with accuracy, but GemOCR takes it to the next level, using AI models to clean up the chaos and deliver pristine, digital text from old manuals. Motivation to make this must go to GilgameshofUT, creator of AI Dungeon Delver (https://github.com/GilgameshofUT/AIDungeon-Delver), this project gives those forgotten sourcebooks a new lease of life so you can shove them into that project.  The World of Darkness of my youth beckons!  

🔥 Features
OCR Meets AI: Combines Google Cloud Vision OCR with Gemini and Cohere AI models to smartly clean, format, and restore text with minimal errors.
Seamless Exports: Automatically converts your processed files into Markdown and PDF formats for ultimate flexibility.
Image Preprocessing: Prepares your documents with enhanced image clarity to ensure superior OCR accuracy.
Failsafe AI Switching: Automatically switches between AI models if resources run low, ensuring consistent results without interruptions.
Rate-limited for Stability: Built-in protection against overloading APIs, making sure your work flows smoothly.
🚀 Installation
Getting started with GemOCR is a breeze. Here’s how:

Clone the Repository:

```bash
Copy code
git clone https://github.com/username/GemOCR.git
```

Install Dependencies:

```bash
pip install -r requirements.txt
```
Set Up Your API Keys: Create a .env file in the project root or set enviroment variables, with the following:
GEMINI_API_KEY=your_gemini_api_key
COHERE_API_KEY=your_cohere_api_key


Run the Project: Just fire it up with:

```bash
python main.py
```

⚙️ How It Works
Here’s the magic behind GemOCR:

Text Extraction: We first attempt to pull text directly from your PDFs using PyPDF2. If it fails, GemOCR rolls up its sleeves and runs high-precision OCR on each page.

AI-Powered Cleanup: The extracted text is processed through AI models, fixing the OCR’s quirks, formatting errors, and creating a polished, clean document. We use Gemini first, but if things get tight, Cohere steps in to handle the load.

Consistent Formatting: All extracted text is formatted consistently, cleaned of redundant spaces, and styled properly using Markdown syntax. Your sourcebooks will be tidy and professional.

Dual Format Output: Output is saved as both Markdown and PDF, ready for any use. Whether you want to edit in a text editor or print a slick PDF—GemOCR has you covered.

📂 Usage
Drop Your PDFs: Put your RPG sourcebooks in the Input folder.
Run the Processor:
bash
Copy code
python main.py
Collect Your Output: Find the digitised, cleaned versions of your sourcebooks in the Output folder, both as Markdown (.md) and PDF (.pdf). The original PDFs are safely moved to the Processed folder.
🔮 What Makes GemOCR Special?
This isn’t just OCR. GemOCR doesn’t settle for mediocre text recognition. It processes your data with a mix of OCR and AI cleanup to deliver high-quality, error-free documents that are ready to use—whether you're diving into RPGs, building digital archives, or setting up content for AI-driven adventures.

🛠️ Licence
GemOCR is open-source. Use it, share it, and modify it as you please!

🧠 Acknowledgements
A huge shout-out to GilgameshofUT for creating the original AI Dungeon Delver, which sparked the inspiration behind GemOCR.

⚡ Final Thoughts
While there are always faster ways to get things done, GemOCR keeps things smooth and stable. We’ve balanced speed with reliability—so even if it’s not pushing warp speeds, it gets you the quality result every time. Feel like tweaking it? Dive into the code and make it even better—we welcome your contributions!
