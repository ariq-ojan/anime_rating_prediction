# **📺 Anime Rating Prediction with Machine Learning**

Most user generated ratings from streaming platforms are noisy, biased, and unreliable. Popularity spikes can inflate scores, and older, less-known titles have only a few votes. This inconsistencies hurt recommendation quality and content discovery for users. We aim to develop a more reliable and data-driven model to estimate expected audience score of animes. The insights can be used by platforms to improve their catalog management, content recommendations for users, and allocate marketing budget.

**Goal**: Accurately estimate an anime's audience score (1-10) to improve content ranking and user recommendation.

**Analytical approach**: Develop a machine learning model that predicts the expected audience rating of an anime using intrinsic (genre, type, duration, studio, etc) and engagement (favorites, member amounts, popularity, etc) attributes.

**Stakeholder**: Streaming platform product managers.

## **Objectives**
1. Develop a machine learning model that predicts expected user score with high accuracy (RMSE < 0.5).
2. Provide feature insights into what drives audience perception.

## **Evaluation Metrics**
- **RMSE** (target < 0.5): Perfect for models that wants to achieve high accuracy.
- **R² score**: Captures proportion of variance and model stability.
- **MAPE**: Easily interpretable percentage metric for stakeholders.

## **Executive Summary**
- The developed model was able to successfully predict anime scores with high accuracy and performance **(RMSE 0.375, and MAPE 4.37%)**.
- Audience engagement metrics **(favorites, engagement ratio, and scored_by)** are the most influential indicators of how highly an anime is rated.
- Model can be used to **prioritize marketing budget** on possible hit titles to maximize return.

----

## **Data Dictionary**
Data used is the Top 15,000 Ranked Anime Dataset that was collected from MyAnimeList by Quanthan which can be found [here](https://www.kaggle.com/datasets/quanthan/top-15000-ranked-anime-dataset-update-to-32025). Each row represents a unique anime and they are ordered by their MyAnimeList rank.

| Column          | Description |
|-----------------|-------------|
| anime_id        | Unique identifier for the anime on MyAnimeList (MAL ID). |
| anime_url       | URL link to the anime's page on MyAnimeList. |
| image_url       | URL of the anime's main visual or cover image (JPEG format). |
| name            | Official title of the anime. |
| english_name    | Official English title of the anime, if available. |
| japanese_names  | Official Japanese title of the anime, if available. |
| score           | Average score/rating of the anime on MyAnimeList (1–10, higher is better). |
| genres          | Comma-separated list of genres associated with the anime (e.g., Action, Comedy, Fantasy). |
| themes          | Comma-separated list of themes associated with the anime (e.g., Psychological, Time Travel). |
| demographics    | Target audience of the anime (e.g., Shounen, Shoujo, Seinen, Josei). |
| synopsis        | Brief summary or plot description of the anime. |
| type            | Type of anime (e.g., TV, Movie, OVA, ONA, Special, Music). |
| episodes        | Number of episodes. |
| premiered       | Season and year the anime premiered (e.g., "Fall 2013"). |
| producers       | Comma-separated list of production companies involved. |
| studios         | Animation studios responsible for creating the anime. |
| source          | Original source material (e.g., Manga, Original, Light Novel, Game). |
| duration        | Duration of each episode in minutes. |
| rating          | Age/content rating (e.g., PG-13, R+, G). |
| rank            | MAL ranking based on score (lower is better). Dataset contains the top ~15k anime. |
| popularity      | Popularity rank on MAL (lower is more popular). |
| favorites       | Number of users who marked this anime as a favorite. |
| scored_by       | Number of users who rated the anime. |
| members         | Number of users listed as members of the anime’s MAL community. |

## How to Use
You can the deployed web app for this project [here!](http://malratingpredictorojan2.streamlit.app)
1. Use the search box to find your anime on MyAnimeList.
2. From the anime page, copy the Members count and enter it into this tool.
3. Fill in the remaining fields using information from the Information and Statistics sections.
4. Click "Predict" to generate estimated user score.

### ⚠️ Model Limitation
- Currently limited to only MyAnimeList data and statistics.
- Model will not accurately estimate low-rated scores (score < 5).

----
© 2025 Ariq Fauzan & Fauzan Hafizh  
This project was designed, developed, and implemented by Ariq Fauzan and Fauzan Hafizh.
All data used belongs to their respective owners. You’re free to use, modify, and distribute this project with proper attribution.

Email:
[mariqfauzan@gmail.com](mailto:mariqfauzan@gmail.com)
[fhafizh2710@gmail.com](mailto:fhafizh2710@gmail.com)  
LinkedIn:
[Ariq Fauzan](https://www.linkedin.com/in/ariq-fauzan/)
[Fauzan Hafizh](https://www.linkedin.com/in/fauzan-hafizh-2678a5252/?utm_source=share_via&utm_content=profile&utm_medium=member_ios)
