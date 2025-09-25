# Dataset Instructions: Flickr8k

This project uses the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) for image captioning.

## 1. Download the dataset
You can download it from Kaggle:

1. Go to the Kaggle page: [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
2. Click **Download** → you will get a zip file containing:
   - `Flickr8k_Dataset/Images/` → 8,000 images
   - `Flickr8k_text/Flickr8k.token.txt` → captions file

> Alternatively, you can use Kaggle API to download directly:
```bash
kaggle datasets download -d adityajn105/flickr8k
unzip flickr8k.zip -d data/
