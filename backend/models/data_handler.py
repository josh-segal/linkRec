import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from typing import List, Dict, Tuple, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self, data_path: str):
        """
        Initialize DataHandler with path to MIND dataset
        
        Args:
            data_path: Path to directory containing MINDsmall_train
        """
        self.data_path = data_path
        self.news_df = None
        self.behaviors_df = None
        self.news_features = None
        self.scaler = None
        self.feature_dim = None
        self.processed_data = None

    def load_data(self, sample_fraction: float = 1.0, random_state: Optional[int] = None) -> None:
        """
        Load and preprocess the MIND dataset
        """
        # Load news data
        logger.info("Loading news data...")
        col_news = ['NewsId', 'Category', 'SubCat', 'Title', 'Abstract', 'url', 'TitleEnt', 'AbstractEnt']
        self.news_df = pd.read_csv(
            os.path.join(self.data_path, 'MINDsmall_train/news.tsv'),
            sep='\t', header=None, names=col_news
        )
        
        # Load behaviors data
        logger.info("Loading behaviors data...")
        col_behaviors = ['ImpressionID', 'UserID', 'Time', 'History', 'Impressions']
        self.behaviors_df = pd.read_csv(
            os.path.join(self.data_path, 'MINDsmall_train/behaviors.tsv'),
            sep='\t', header=None, names=col_behaviors
        )
        
        # Sample if requested
        if sample_fraction < 1.0:
            self.behaviors_df = self.behaviors_df.sample(frac=sample_fraction, random_state=random_state)
        
        # Process data
        logger.info("Encoding categorical features...")
        self.news_df = self._encode_categorical_features(self.news_df)
        
        logger.info("Preparing bandit data...")
        self.processed_data, self.news_features = self._prepare_data_for_bandit()
        
        logger.info("Creating feature scaler...")
        self.scaler = self._create_feature_scaler()
        
        self.feature_dim = 4 + len(self.news_features.columns)  # 4 context features + news features
        logger.info(f"Data loading complete. Feature dimension: {self.feature_dim}")

    def _encode_categorical_features(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using transformer-based clustering
        """
        logger.info("Loading sentence transformer model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get unique categories and subcategories
        categories = news_df['Category'].unique()
        subcategories = news_df['SubCat'].unique()
        
        # Create embeddings
        logger.info("Creating embeddings for categories and subcategories...")
        category_embeddings = model.encode(categories)
        subcat_embeddings = model.encode(subcategories)
        
        # Perform K-means clustering
        n_cat_clusters = min(len(categories), 10)
        n_subcat_clusters = min(len(subcategories), 20)
        
        logger.info("Clustering categories...")
        cat_kmeans = KMeans(n_clusters=n_cat_clusters, random_state=42)
        subcat_kmeans = KMeans(n_clusters=n_subcat_clusters, random_state=42)
        
        cat_clusters = cat_kmeans.fit_predict(category_embeddings)
        subcat_clusters = subcat_kmeans.fit_predict(subcat_embeddings)
        
        # Create mappings
        cat_to_cluster = dict(zip(categories, cat_clusters))
        subcat_to_cluster = dict(zip(subcategories, subcat_clusters))
        
        # Add cluster assignments
        cluster_df = pd.DataFrame({
            'category_cluster': news_df['Category'].map(cat_to_cluster),
            'subcategory_cluster': news_df['SubCat'].map(subcat_to_cluster)
        })
        
        # Convert clusters to one-hot encoding
        cluster_cat_ohe = pd.get_dummies(cluster_df['category_cluster'], prefix='cluster_cat')
        cluster_subcat_ohe = pd.get_dummies(cluster_df['subcategory_cluster'], prefix='cluster_subcat')
        
        return pd.concat([news_df, cluster_cat_ohe, cluster_subcat_ohe], axis=1)

    def _extract_time_features(self, time_str: str) -> Dict:
        """
        Extract time-based features from timestamp
        """
        time = datetime.strptime(time_str, '%m/%d/%Y %I:%M:%S %p')
        return {
            'hour': time.hour,
            'day_of_week': time.weekday(),
            'is_weekend': 1 if time.weekday() >= 5 else 0
        }

    def _prepare_data_for_bandit(self) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Prepare data in format suitable for bandit
        """
        # Extract relevant features
        category_cols = [col for col in self.news_df.columns if col.startswith('cluster_cat_')]
        subcategory_cols = [col for col in self.news_df.columns if col.startswith('cluster_subcat_')]
        
        # Combine features
        news_features = pd.concat([
            self.news_df[['NewsId']],
            self.news_df[category_cols],
            self.news_df[subcategory_cols]
        ], axis=1).set_index('NewsId')
        
        interactions = []
        total_rows = len(self.behaviors_df)
        valid_rows = 0
        
        logger.info("Processing behavior data...")
        for _, row in self.behaviors_df.iterrows():
            # Extract time features
            time_features = self._extract_time_features(row['Time'])
            
            # Create context
            context = {
                'history_len': len(row['History'].split()) if pd.notna(row['History']) else 0,
                **time_features
            }
            
            # Process impressions
            impressions = row['Impressions'].split()
            slate = []
            rewards = []
            
            for imp in impressions:
                news_id, click = imp.split('-')
                if news_id in news_features.index:
                    slate.append(news_id)
                    rewards.append(int(click))
            
            if slate:  # Only include if there are valid articles
                valid_rows += 1
                interactions.append({
                    'user_id': row['UserID'],
                    'context': context,
                    'slate': slate,
                    'rewards': rewards,
                    'news_features': news_features.loc[slate].to_dict('records')
                })
        
        logger.info(f"Total behavior rows: {total_rows}")
        logger.info(f"Valid interactions created: {valid_rows}")
        logger.info(f"Percentage of valid interactions: {(valid_rows/total_rows)*100:.2f}%")
        
        return interactions, news_features

    def prepare_features_vector(self, context: Dict, news_feat: Dict) -> np.ndarray:
        """
        Combine context and news features into a single vector
        """
        context_vec = np.array([
            context['history_len'],
            context['hour'],
            context['day_of_week'],
            context['is_weekend']
        ])
        
        news_vec = np.array([v for k, v in news_feat.items() if k != 'NewsId'])
        return np.concatenate([context_vec, news_vec])

    def _create_feature_scaler(self) -> StandardScaler:
        """
        Create and fit StandardScaler for feature vectors
        """
        all_feature_vectors = []
        
        for interaction in self.processed_data:
            context = interaction['context']
            for news_feat in interaction['news_features']:
                features = self.prepare_features_vector(context, news_feat)
                all_feature_vectors.append(features)
        
        scaler = StandardScaler()
        scaler.fit(np.array(all_feature_vectors))
        return scaler

    def get_random_context(self) -> Dict:
        """
        Sample a random context from the dataset
        """
        # Get a random index instead of using np.random.choice directly on the list
        random_idx = np.random.randint(0, len(self.processed_data))
        interaction = self.processed_data[random_idx]
        
        # Log more details about the selected interaction
        logger.info(f"Selected interaction {random_idx} of {len(self.processed_data)}")
        logger.info(f"Context: {interaction['context']}")
        logger.info(f"Number of articles in slate: {len(interaction['news_features'])}")
        
        return {
            'context': interaction['context'],
            'articles': interaction['news_features'],
            'true_rewards': interaction['rewards']
        }

    def get_scaled_features(self, context: Dict, news_feat: Dict) -> np.ndarray:
        """
        Get scaled feature vector for context-article pair
        """
        features = self.prepare_features_vector(context, news_feat)
        return self.scaler.transform(features.reshape(1, -1))[0]

    def get_feature_dim(self) -> int:
        """Return the dimension of feature vectors"""
        return self.feature_dim