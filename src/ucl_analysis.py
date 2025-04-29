import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

class UCLAnalyzer:
    def __init__(self, performance_path, finals_path):
        self.performance_path = performance_path
        self.finals_path = finals_path
        self.df = None
        self.finals_df = None
        self.model = None
        self.load_data()

    def load_data(self):
        """Chargement et préparation des données"""
   # Chargement des données de performance générale
        self.df = pd.read_csv(self.performance_path)
        # Nettoyage et traitement avancé des colonnes de buts
        self.df['goals_scored'] = self.df['goals'].apply(lambda x: int(x.split(':')[0]))
        self.df['goals_conceded'] = self.df['goals'].apply(lambda x: int(x.split(':')[1].split(':')[0]))
        
        # Calcul des ratios de performance
        self.df['win_ratio'] = self.df['W'] / self.df['M.']
        self.df['draw_ratio'] = self.df['D'] / self.df['M.']
        self.df['loss_ratio'] = self.df['L'] / self.df['M.']
        self.df['points_per_game'] = self.df['Pt.'] / self.df['M.']
        
        # Efficacité offensive et défensive
        self.df['goals_per_game'] = self.df['goals_scored'] / self.df['M.']
        self.df['goals_conceded_per_game'] = self.df['goals_conceded'] / self.df['M.']
        self.df['score_efficiency'] = (self.df['goals_scored'] - self.df['goals_conceded']) / self.df['M.']
        
        # Performance historique
        self.df['win_percentage'] = (self.df['W'] / self.df['M.'] * 100).round(2)
        self.df['goal_difference_per_game'] = self.df['Dif'] / self.df['M.']
        self.df['clean_sheets_ratio'] = (self.df['goals_conceded'] == 0).sum() / self.df['M.']
        
        # Chargement des données des finales avec encodage UTF-8
        self.finals_df = pd.read_csv(self.finals_path, encoding='utf-8')
        
        # Nettoyage des scores
        self.finals_df['Score'] = self.finals_df['Score'].str.replace('â€"', '-')
        
        # Extraction des buts pour les finales
        self.finals_df[['Winner_Goals', 'Runner_Goals']] = self.finals_df['Score'].str.extract(r'(\d+)[^\d]+(\d+)')
        self.finals_df[['Winner_Goals', 'Runner_Goals']] = self.finals_df[['Winner_Goals', 'Runner_Goals']].astype(int)
        
        # Ajout d'une colonne pour les matchs avec prolongation
        self.finals_df['Extra_Time'] = self.finals_df['Notes'].str.contains('extra time', case=False, na=False)
        
        # Calcul des statistiques des finales par équipe
        finals_stats = self.calculate_finals_stats()

    def analyze_historical_trends(self):
        """Analyse des tendances historiques"""        # Performance globale des équipes
        top_teams = self.df.nlargest(10, 'points_per_game').copy()
        
        # Visualisation des points par match
        plt.figure(figsize=(15, 8))
        sns.barplot(data=top_teams, x='Team', y='points_per_game')
        plt.title('Top 10 des Équipes par Points par Match en UCL')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('docs/points_per_game.svg')
        plt.close()

        # Efficacité offensive et défensive
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=self.df.head(20), x='goals_per_game', y='win_percentage', size='M.', 
                       sizes=(100, 1000), alpha=0.7)
        plt.title('Buts Marqués par Match vs Pourcentage de Victoires')
        
        plt.subplot(1, 2, 2)
        sns.scatterplot(data=self.df.head(20), x='goals_conceded_per_game', y='loss_ratio', size='M.', 
                       sizes=(100, 1000), alpha=0.7)
        plt.title('Buts Encaissés par Match vs Ratio de Défaites')
        plt.tight_layout()
        plt.savefig('docs/team_efficiency.svg')
        plt.close()

        # Statistiques par pays
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.country_stats.head(10), x='Country', y='country_titles')
        plt.title('Top 10 des Pays par Nombre de Titres en UCL')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('docs/country_performance.svg')
        plt.close()

        # Statistiques des stades
        plt.figure(figsize=(12, 6))
        sns.barplot(data=self.venue_stats.head(10), x='times_hosted', y='Venue')
        plt.title('Top 10 des Stades Ayant Accueilli le Plus de Finales')
        plt.tight_layout()
        plt.savefig('docs/venue_statistics.svg')
        plt.close()

        # Analyse des victoires en prolongation
        top_et_teams = self.df.nlargest(10, 'extra_time_total').copy()
        
        # Création d'un graphique à barres empilées
        bar_width = 0.35
        index = np.arange(len(top_et_teams))
        
        plt.bar(index, top_et_teams['extra_time_wins'], bar_width,
                label='Victoires', color='#2ecc71')
        plt.bar(index, top_et_teams['extra_time_total'] - top_et_teams['extra_time_wins'],
                bar_width, bottom=top_et_teams['extra_time_wins'],
                label='Défaites', color='#e74c3c')
        
        # Ajout des pourcentages de victoire
        for i, team in enumerate(top_et_teams['Team']):
            win_ratio = top_et_teams.iloc[i]['extra_time_win_ratio'] * 100
            plt.text(i, top_et_teams.iloc[i]['extra_time_total'] + 0.1,
                     f'{win_ratio:.0f}%', ha='center')
        
        plt.xlabel('Équipes')
        plt.ylabel('Nombre de Matches en Prolongation')
        plt.title('Performance des Équipes en Prolongation')
        plt.xticks(index, top_et_teams['Team'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig('docs/extra_time_wins.svg')
        plt.close()

        # Moyenne de buts en finale
        top_scoring = self.df.nlargest(10, 'avg_goals_scored')
        plt.figure(figsize=(12, 6))
        sns.barplot(data=top_scoring, x='Team', y='avg_goals_scored')
        plt.title('Top 10 des Équipes par Moyenne de Buts en Finale')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('docs/average_goals_finals.svg')
        plt.close()

    def calculate_finals_stats(self):
        """Calcul des statistiques des finales pour chaque équipe"""
        # Statistiques par équipe
        winners = self.finals_df['Winners'].value_counts().reset_index()
        winners.columns = ['Team', 'finals_won']
        runners = self.finals_df['Runners-up'].value_counts().reset_index()
        runners.columns = ['Team', 'finals_lost']
        
        # Statistiques par pays
        country_winners = self.finals_df['Country'].value_counts().reset_index()
        country_winners.columns = ['Country', 'country_titles']
        
        # Statistiques des stades
        venue_stats = self.finals_df['Venue'].value_counts().reset_index()
        venue_stats.columns = ['Venue', 'times_hosted']
        
        # Statistiques avancées par équipe
        finals_stats = pd.merge(winners, runners, on='Team', how='outer').fillna(0)
        finals_stats['finals_total'] = finals_stats['finals_won'] + finals_stats['finals_lost']
        finals_stats['finals_win_ratio'] = finals_stats['finals_won'] / finals_stats['finals_total']
        
        # Calcul amélioré des victoires en prolongation
        extra_time_matches = self.finals_df[self.finals_df['Extra_Time'] == True]
        finals_stats['extra_time_wins'] = extra_time_matches['Winners'].value_counts().reindex(finals_stats['Team']).fillna(0)
        finals_stats['extra_time_total'] = finals_stats['extra_time_wins'] + \
            extra_time_matches[extra_time_matches['Runners-up'].isin(finals_stats['Team'])]['Runners-up'].value_counts().reindex(finals_stats['Team']).fillna(0)
        finals_stats['extra_time_win_ratio'] = (finals_stats['extra_time_wins'] / finals_stats['extra_time_total']).fillna(0)
        
        # Calcul des moyennes de buts
        finals_stats['avg_goals_scored'] = 0.0
        for team in finals_stats['Team']:
            wins_goals = self.finals_df[self.finals_df['Winners'] == team]['Winner_Goals'].mean()
            runner_goals = self.finals_df[self.finals_df['Runners-up'] == team]['Runner_Goals'].mean()
            finals_stats.loc[finals_stats['Team'] == team, 'avg_goals_scored'] = (
                pd.concat([self.finals_df[self.finals_df['Winners'] == team]['Winner_Goals'],
                          self.finals_df[self.finals_df['Runners-up'] == team]['Runner_Goals']]).mean()
            )
        
        # Fusion avec le DataFrame principal
        self.df = pd.merge(self.df, 
                          finals_stats[['Team', 'finals_won', 'finals_lost', 'finals_total', 
                                      'finals_win_ratio', 'extra_time_wins', 'avg_goals_scored']], 
                          on='Team', how='left').fillna(0)
        
        # Stockage des statistiques supplémentaires
        self.country_stats = country_winners
        self.venue_stats = venue_stats
        
        return finals_stats

    def train_prediction_model(self):
        """Entraînement du modèle de prédiction"""
        # Préparation des features avec les statistiques des finales
        features = ['M.', 'W', 'D', 'L', 'goals_scored', 'goals_conceded', 'win_ratio', 'score_efficiency']
        X = self.df[features].fillna(0)
        y = (self.df['Pt.'] > self.df['Pt.'].median()).astype(int)  # Classification binaire

        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalisation
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Entraînement du modèle
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Évaluation du modèle
        accuracy = self.model.score(X_test_scaled, y_test)
        print(f'Précision du modèle: {accuracy:.2f}')

        # Importance des features
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Importance des Caractéristiques pour la Prédiction')
        plt.tight_layout()
        plt.savefig('docs/feature_importance.svg')
        plt.close()

    def predict_future_winners(self, top_n=5):
        """Prédiction des équipes ayant le plus de chances de gagner"""
        if self.model is None:
            self.train_prediction_model()

        # Préparation des données pour la prédiction
        features = ['M.', 'W', 'D', 'L', 'goals_scored', 'goals_conceded', 'win_ratio', 'score_efficiency']
        X = self.df[features].fillna(0)
        
        # Normalisation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Probabilités de victoire
        win_probs = self.model.predict_proba(X_scaled)[:, 1]
        
        # Création d'un DataFrame avec les probabilités
        predictions = pd.DataFrame({
            'Team': self.df['Team'],
            'Win_Probability': win_probs
        }).sort_values('Win_Probability', ascending=False)
        
        return predictions.head(top_n)

def main():
    # Initialisation de l'analyseur avec les deux sources de données
    analyzer = UCLAnalyzer('data/UCL_AllTime_Performance_Table.csv', 'data/UCL_Finals_1955-2023.csv')
    
    # Analyse des tendances historiques
    print('Analyse des tendances historiques...')
    analyzer.analyze_historical_trends()
    
    # Entraînement du modèle et prédictions
    print('\nEntraînement du modèle de prédiction...')
    analyzer.train_prediction_model()
    
    # Prédiction des futurs vainqueurs potentiels
    print('\nPrédiction des futurs vainqueurs potentiels:')
    predictions = analyzer.predict_future_winners()
    print(predictions)

if __name__ == '__main__':
    main()