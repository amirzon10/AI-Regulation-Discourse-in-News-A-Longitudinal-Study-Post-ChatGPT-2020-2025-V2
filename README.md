# AI Regulation Discourse in News: A Longitudinal Study Post-ChatGPT (2020-2025)

## 📋 Project Description

This repository contains a comprehensive longitudinal analysis of AI regulation discourse in international news media from 2020 to 2025. The project examines how news coverage of AI regulation has evolved, particularly following the release of ChatGPT, by analyzing sentiment patterns, named entities, and political orientation across multiple languages and news sources.

The study provides insights into:
- **Temporal trends** in AI regulation discourse
- **Sentiment patterns** toward AI regulation across different media outlets
- **Named entities** frequently associated with AI regulation discussions
- **Political orientation** of news sources covering AI regulation
- **Cross-linguistic analysis** of AI regulation narratives

## 🎯 Research Objectives

- Analyze the evolution of AI regulation discourse in global news media (2020-2025)
- Identify sentiment patterns and changes over time, especially post-ChatGPT
- Extract and analyze key entities (organizations, people, locations) in AI regulation discussions
- Compare discourse patterns across different political orientations and regions
- Provide data-driven insights for policymakers, researchers, and journalists

## 📁 Repository Structure

```
.
├── data/
│   ├── Datasets contributions/
│   │   └── orientaion_english.csv          # Political orientation metadata
│   ├── analysis.csv                         # Main analysis results
│   ├── dataset_combined_pretranslation_taba.csv  # Combined pre-translation dataset
│   ├── european_dataset.csv                 # European news sources data
│   ├── media_cloud-political_orientation.csv  # Media Cloud with political labels
│   ├── mediacloud-translated-complete_part1.csv  # Translated articles (part 1)
│   ├── mediacloud-translated-complete_part2.csv  # Translated articles (part 2)
│   ├── mediacloud-translated-completed_taba.csv  # Complete translated dataset
│   ├── sentiment-analysis.csv               # Sentiment analysis results
│   ├── translate_part1.csv                  # Translation batch 1
│   └── translate_part2.csv                  # Translation batch 2
│
├── data_engineering/
│   ├── preprocessing.ipynb                  # Data preprocessing and filtering
│   ├── language_translation.ipynb           # Multi-language translation pipeline
│   ├── merge_datasets.ipynb                 # Dataset merging and integration
│   └── political_orientation.ipynb          # Political orientation labeling
│
└── Sentiment_analysis_and_NER/
    ├── sentiment_analysis_Vader.ipynb       # VADER sentiment analysis
    ├── sentiment_analysis_RoBERTa.ipynb     # RoBERTa-based sentiment analysis
    ├── NER_EN_CORE.ipynb                    # Named Entity Recognition (spaCy)
    └── NER_RoBERTa.ipynb                    # Named Entity Recognition (RoBERTa)
```

## 🔬 Methodology

### Data Collection
- **Source**: MediaCloud database (2020-2025)
- **Languages**: Multiple languages including Chinese (zh), Japanese (ja), English (en), Indonesian (id), Spanish (es), French (fr), Portuguese (pt), Korean (ko), and 20+ others
- **Focus**: News articles discussing AI regulation and governance

### Data Processing Pipeline

1. **Preprocessing** (`data_engineering/preprocessing.ipynb`)
   - Load combined dataset from multiple sources
   - Inspect language distribution
   - Filter to Western languages using configurable language sets
   - Split data for translation processing
   - Visualize language counts and distribution

2. **Language Translation** (`data_engineering/language_translation.ipynb`)
   - Translate non-English articles to English using Google Translator
   - Chunked processing for large-scale multilingual datasets (default: 1000 rows)
   - Preserve original English titles
   - Robust error handling for translation failures

3. **Dataset Merging** (`data_engineering/merge_datasets.ipynb`)
   - Integrate multiple data sources
   - Combine European and MediaCloud datasets
   - Resolve duplicates and inconsistencies

4. **Political Orientation Labeling** (`data_engineering/political_orientation.ipynb`)
   - Assign political orientation labels to news sources
   - Categorize sources by political leaning

### Analysis Methods

#### Sentiment Analysis
Two complementary approaches are implemented:

1. **VADER Lexicon-Based Analysis** (`sentiment_analysis_Vader.ipynb`)
   - Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - Lexicon-based approach suitable for social media and news text
   - Provides compound sentiment scores

2. **RoBERTa Deep Learning Analysis** (`sentiment_analysis_RoBERTa.ipynb`)
   - Transformer-based sentiment classification
   - Fine-tuned RoBERTa model for news sentiment
   - Feature extraction using TF-IDF and Truncated SVD (LSA)
   - Handles constraint of no labeled training data

#### Named Entity Recognition (NER)
Two NER approaches for comprehensive entity extraction:

1. **spaCy EN_CORE** (`NER_EN_CORE.ipynb`)
   - Uses spaCy's English core model
   - Extracts organizations, persons, locations, dates
   - Fast and efficient for large-scale processing

2. **RoBERTa-based NER** (`NER_RoBERTa.ipynb`)
   - Fine-tuned transformer model for entity recognition
   - Higher accuracy for domain-specific entities
   - Captures nuanced entity mentions in AI regulation context

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **pandas**: Data manipulation and analysis
- **deep-translator**: Multi-language translation via Google Translator
- **transformers (Hugging Face)**: RoBERTa models for NLP tasks
- **scikit-learn**: Machine learning utilities (TF-IDF, TruncatedSVD)
- **spaCy**: NLP pipeline for entity recognition
- **VADER**: Sentiment analysis lexicon
- **matplotlib & seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment

## 📊 Key Features

- **Multilingual Processing**: Handles 20+ languages with automated translation
- **Temporal Analysis**: Tracks discourse evolution from 2020-2025
- **Dual Sentiment Analysis**: Combines lexicon-based and deep learning approaches
- **Comprehensive NER**: Identifies key actors and entities in AI regulation discourse
- **Political Context**: Analyzes discourse variation across political orientations
- **Scalable Pipeline**: Chunked processing for large datasets
- **Reproducible Research**: Well-documented notebooks with clear methodology

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy
pip install deep-translator
pip install transformers torch
pip install scikit-learn
pip install spacy
pip install vaderSentiment
pip install matplotlib seaborn

# Download spaCy English model
python -m spacy download en_core_web_sm
```

### Usage

1. **Data Preprocessing**
   ```bash
   # Run preprocessing notebook
   jupyter notebook data_engineering/preprocessing.ipynb
   ```

2. **Translation Pipeline** (if working with non-English data)
   ```bash
   jupyter notebook data_engineering/language_translation.ipynb
   ```

3. **Sentiment Analysis**
   ```bash
   # VADER analysis
   jupyter notebook Sentiment_analysis_and_NER/sentiment_analysis_Vader.ipynb
   
   # RoBERTa analysis
   jupyter notebook Sentiment_analysis_and_NER/sentiment_analysis_RoBERTa.ipynb
   ```

4. **Named Entity Recognition**
   ```bash
   # spaCy NER
   jupyter notebook Sentiment_analysis_and_NER/NER_EN_CORE.ipynb
   
   # RoBERTa NER
   jupyter notebook Sentiment_analysis_and_NER/NER_RoBERTa.ipynb
   ```

## 📈 Expected Outputs

- **Sentiment trends** over time (2020-2025)
- **Entity frequency analysis** (top organizations, people, locations)
- **Political orientation comparisons** in AI regulation coverage
- **Language-specific discourse patterns**
- **Visualizations** of temporal and categorical trends

## 🔍 Research Applications

This dataset and analysis framework can support:
- Academic research on AI governance and policy
- Media studies on technology coverage
- Policy analysis and regulatory development
- Comparative studies across regions and political contexts
- Temporal analysis of public discourse evolution

## 📝 Data Sources

- **MediaCloud**: Primary source for news articles (2020-2025)
- **European News Sources**: Supplementary dataset for European coverage
- **Political Orientation Data**: Manually curated and third-party sources

## ⚠️ Limitations

- Translation quality may vary for complex technical terminology
- Sentiment analysis on translated text may not capture all nuances
- Political orientation labels are approximate and may not reflect all aspects
- Dataset coverage may vary across languages and regions
- Some languages might have limited representation. The dataset includes the following language distribution:
  - English (en): 58,184 articles
  - Spanish (es): 22,557 articles
  - Portuguese (pt): 12,072 articles
  - Turkish (tr): 3,086 articles
  - Italian (it): 2,138 articles
  - French (fr): 1,663 articles
  - Ukrainian (uk): 850 articles
  - Russian (ru): 819 articles
  - Dutch (nl): 804 articles
  - German (de): 766 articles
  - Polish (pl): 158 articles

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report issues or bugs
- Suggest new analysis approaches
- Contribute additional datasets
- Improve documentation
- Submit pull requests

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 👥 Authors

- **Amir Freer Valdez** - Primary researcher and developer
- - **Vrynsiaa** - Contributor and co-developer

## 📧 Contact

For questions, suggestions, or collaborations, please open an issue in this repository.

## 🙏 Acknowledgments

- MediaCloud for providing access to news data
- Open-source NLP community for tools and models
- Contributors to the dataset and analysis

```

---

**Last Updated**: December 2025
