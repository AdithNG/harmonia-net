# 🧠 Techniques & Methods Used in HarmoniaNet

---

## 📊 1. Exploratory Data Analysis (EDA)

- **Correlation Heatmap**  
  Used `seaborn` to visualize relationships between numerical features like `energy`, `acousticness`, `tempo`, etc.  
  Helped identify which features are most related and whether any were redundant.

- **Genre-Wise Aggregation**  
  Grouped the dataset by `track_genre` to compute the average popularity and other features for each genre.  
  Visualized using a bar chart to understand how genres differ on average.

---

## 🧹 2. Data Preprocessing

- **Top 10 Genre Selection**  
  Simplified the classification task by selecting the 10 most frequent genres in the dataset.

- **Label Encoding**  
  Converted genre names to numeric labels using `LabelEncoder` for compatibility with PyTorch models.

- **Feature Normalization**  
  Scaled numerical features using `StandardScaler` to ensure they were on the same scale and improve training stability.

- **Train-Test Split (Stratified)**  
  Split the data using `train_test_split` with `stratify=y` to maintain equal genre representation across training and testing sets.

---

## 🔧 3. PyTorch Model Building

- **Neural Network Architecture (`HarmoniaNet`)**

  - Input layer → 256 hidden units
  - Hidden layer → 512 hidden units
  - ReLU activation functions
  - Dropout layers (0.3) to prevent overfitting
  - Output layer with 10 neurons (for 10 genres)

- **GPU Acceleration**  
  Detected and used GPU with `torch.device("cuda")` to speed up training.

---

## 🏋️ 4. Model Training

- **Loss Function**: `CrossEntropyLoss`  
  Suitable for multi-class classification problems.

- **Optimizer**: `Adam`  
  Chosen for its adaptive learning rate and efficient convergence.

- **Epoch-Based Training Loop**  
  Trained the model for 500 epochs and printed loss per epoch.

- **Training Loss Tracking**  
  Collected loss values in a list and plotted a **training loss curve** using `matplotlib` to visualize model convergence.

---

## 🧪 5. Evaluation Techniques

- **Accuracy**  
  Computed the proportion of correct predictions on the test set.

- **Classification Report**  
  Used `sklearn.metrics.classification_report` to get precision, recall, and F1-score per genre.

- **Confusion Matrix**  
  Visualized with a heatmap using `seaborn` to identify genre-specific strengths and common misclassifications.

---

## 📚 6. Dataset Overview

- The dataset contains **metadata for thousands of Spotify tracks** including:

  - Popularity
  - Tempo
  - Loudness
  - Energy
  - Acousticness
  - Danceability
  - Instrumentalness
  - Speechiness
  - Liveness
  - Valence
  - Duration

- These features are **audio-derived descriptors**, not raw audio, allowing for fast processing and lightweight modeling.

- We used the **top 10 most common genres** from the dataset to ensure balance and avoid sparsity.

- Each genre had **200 test samples**, making the classification report directly comparable across classes.

---

## 🎧 7. Observations on Key Features

- **Energy vs Loudness**:  
  These two features had a **strong positive correlation (~0.76)**, confirming that louder songs are often more energetic — especially common in genres like **black-metal** and **afrobeat**.

- **Acousticness**:  
  Genres like **acoustic**, **ambient**, and **bluegrass** showed significantly higher average acousticness values, helping the model distinguish them from more produced or electronic genres.

- **Danceability and Tempo**:  
  Genres with rhythmic focus such as **afrobeat** and **brazil** had higher danceability and tempo values, aligning with cultural expectations for these styles.

- **Instrumentalness**:  
  Genres like **ambient** and **bluegrass** tended to score higher here, which likely contributed to their relatively high precision and recall.

- **Valence and Emotion**:  
  Valence, which measures musical positivity, varied widely across genres. **Anime** and **brazil** tended toward higher valence, while **black-metal** was consistently low — this may have helped the model capture emotional tone in genre classification.

---

## 📊 8. Model Evaluation

- After training HarmoniaNet for 500 epochs, the model achieved the following overall results:

  - **Test Accuracy**: **65%**
  - **Macro-Averaged F1 Score**: **0.65**
  - **Weighted-Averaged F1 Score**: **0.65**

- These scores indicate balanced performance across all 10 genres, with no single class dominating model attention.

### 🔹 Best Performing Genres

| Genre           | Precision | Recall   | F1-Score |
| --------------- | --------- | -------- | -------- |
| **Black-metal** | **0.92**  | **0.92** | **0.92** |
| **Ambient**     | **0.83**  | **0.80** | **0.81** |
| **Bluegrass**   | **0.78**  | **0.83** | **0.80** |
| **Afrobeat**    | **0.81**  | **0.74** | **0.77** |

- These genres likely performed well due to distinctive characteristics in features like **energy**, **instrumentalness**, and **acousticness**.
- The confusion matrix confirms that these classes are predicted accurately with minimal overlap with others.

### 🔸 Most Challenging Genres

| Genre           | Precision | Recall   | F1-Score |
| --------------- | --------- | -------- | -------- |
| **Alt-rock**    | **0.30**  | **0.28** | **0.29** |
| **Alternative** | **0.40**  | **0.50** | **0.44** |
| **Blues**       | **0.67**  | **0.49** | **0.57** |

- **Alt-rock and alternative** were frequently confused with each other. For instance:
  - 99 alt-rock tracks were predicted as alternative.
  - 72 alternative tracks were predicted as alt-rock.
- These genres likely have overlapping audio characteristics (mid-range tempo, energy, or valence), making them difficult to distinguish using metadata alone.
- **Blues** showed moderate precision but low recall, suggesting the model tends to underpredict this class or confuses it with similar acoustic genres.

### 🔍 Confusion Matrix Patterns

- The confusion matrix showed **strong diagonal dominance**, especially for black-metal, ambient, and bluegrass — indicating clear genre separation.
- Only a few genres showed heavy off-diagonal overlap, such as alt-rock and alternative.

### 🧠 Insight

These results demonstrate that metadata-based genre classification is feasible, especially for genres with clear audio characteristics. However, more nuanced genres may require **richer feature inputs**, such as spectrograms or lyrics, to improve precision and reduce genre confusion.
