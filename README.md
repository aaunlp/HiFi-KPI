# HIFI-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings

## Datasets
- **[HiFi-KPI](https://huggingface.co/datasets/AAU-NLP/HiFi-KPI)**
- **[HiFi-KPI Lite](https://huggingface.co/datasets/AAU-NLP/hifi-kpi-lite)**

## Models
- [Sequence labelling (1000 most common)](https://huggingface.co/AAU-NLP/BERT-SL1000)
- [Sequence labelling n=1 presentation (1000 most common)](https://huggingface.co/AAU-NLP/Pre-BERT-SL1000)
- [Sequence labelling n=1 calculation (1000 most common)](https://huggingface.co/AAU-NLP/Cal-BERT-SL1000)
- [Sequence labelling HiFi-KPI Lite](https://huggingface.co/AAU-NLP/Lite-BERT-SL)

You can use the template script (`main.py`) to generate a JSON file with your preferred granularity of the iXBRL tags.

### Usage

```bash
python3 main.py --taxonomy "calculation" --iterations 1
