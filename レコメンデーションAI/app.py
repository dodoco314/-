from flask import Flask, render_template_string, request, jsonify
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# グローバル変数
df_movies = None
df_piv = None
rec_model = None

# HTMLテンプレート
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>映画レコメンデーションシステム</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .section-title {
            font-size: 1.8em;
            color: #333;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        .movie-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .movie-card {
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            padding: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
        }
        .movie-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }
        .movie-card.selected {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-color: #667eea;
        }
        .movie-card input[type="checkbox"] {
            display: none;
        }
        .movie-title {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 8px;
        }
        .movie-id {
            color: #666;
            font-size: 0.9em;
        }
        .movie-card.selected .movie-id {
            color: rgba(255,255,255,0.8);
        }
        .button-container {
            text-align: center;
            margin-top: 30px;
        }
        .submit-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 50px;
            font-size: 1.2em;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        }
        .submit-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            box-shadow: none;
        }
        .selected-count {
            text-align: center;
            font-size: 1.2em;
            color: #667eea;
            margin: 20px 0;
            font-weight: bold;
        }
        .result-list {
            list-style: none;
            padding: 0;
        }
        .result-item {
            background: #f8f9fa;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 12px;
            border-left: 5px solid #667eea;
            transition: all 0.3s ease;
        }
        .result-item:hover {
            transform: translateX(5px);
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }
        .result-number {
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            width: 40px;
            height: 40px;
            line-height: 40px;
            text-align: center;
            border-radius: 50%;
            margin-right: 15px;
            font-weight: bold;
        }
        .back-btn {
            display: inline-block;
            background: #6c757d;
            color: white;
            padding: 12px 30px;
            border-radius: 50px;
            text-decoration: none;
            transition: all 0.3s ease;
            margin-top: 20px;
        }
        .back-btn:hover {
            background: #5a6268;
            transform: translateY(-2px);
        }
        .loading {
            text-align: center;
            padding: 40px;
            display: none;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #667eea;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 映画レコメンデーションシステム</h1>
            <p>お好きな映画を3つ選んでください</p>
        </div>
        
        <div class="content">
            <h2 class="section-title">映画を選択してください</h2>
            
            <div class="selected-count" id="selectedCount">
                選択中: <span id="count">0</span> / 3以上
            </div>
            
            <form id="movieForm">
                <div class="movie-grid" id="movieGrid">
                    <!-- 映画カードはJavaScriptで動的生成 -->
                </div>
                
                <div class="button-container">
                    <button type="submit" class="submit-btn" id="submitBtn" disabled>
                        オススメ映画を表示
                    </button>
                </div>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>レコメンデーション計算中...</p>
            </div>
            
            <div id="results" style="display: none;">
                <h2 class="section-title">おすすめ映画トップ5</h2>
                <ul class="result-list" id="resultList"></ul>
                <div class="button-container">
                    <a href="/" class="back-btn">最初に戻る</a>
                </div>
            </div>
        </div>
    </div>

    <script>
        let selectedMovies = new Set();
        let movies = [];

        // 映画リストを取得
        fetch('/api/movies')
            .then(response => response.json())
            .then(data => {
                movies = data;
                renderMovies(movies);
            });

        function renderMovies(movieList) {
            const grid = document.getElementById('movieGrid');
            grid.innerHTML = '';
            
            movieList.forEach(movie => {
                const card = document.createElement('div');
                card.className = 'movie-card';
                card.innerHTML = `
                    <input type="checkbox" id="movie_${movie.id}" value="${movie.id}">
                    <div class="movie-title">${movie.title}</div>
                    <div class="movie-id">ID: ${movie.id}</div>
                `;
                
                card.addEventListener('click', () => {
                    const checkbox = card.querySelector('input[type="checkbox"]');
                    checkbox.checked = !checkbox.checked;
                    
                    if (checkbox.checked) {
                        selectedMovies.add(movie.id);
                        card.classList.add('selected');
                    } else {
                        selectedMovies.delete(movie.id);
                        card.classList.remove('selected');
                    }
                    
                    updateSelectedCount();
                });
                
                grid.appendChild(card);
            });
        }

        function updateSelectedCount() {
            const count = selectedMovies.size;
            document.getElementById('count').textContent = count;
            document.getElementById('submitBtn').disabled = count < 3;
        }

        document.getElementById('movieForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const movieIds = Array.from(selectedMovies);
            
            // ローディング表示
            document.getElementById('loading').classList.add('active');
            document.getElementById('movieForm').style.display = 'none';
            
            try {
                const response = await fetch('/recommend', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ movie_ids: movieIds })
                });
                
                const data = await response.json();
                
                // ローディング非表示
                document.getElementById('loading').classList.remove('active');
                
                // 結果表示
                displayResults(data.recommendations);
            } catch (error) {
                console.error('Error:', error);
                alert('エラーが発生しました');
                document.getElementById('loading').classList.remove('active');
                document.getElementById('movieForm').style.display = 'block';
            }
        });

        function displayResults(recommendations) {
            const resultList = document.getElementById('resultList');
            resultList.innerHTML = '';
            
            recommendations.forEach((movie, index) => {
                const li = document.createElement('li');
                li.className = 'result-item';
                li.innerHTML = `
                    <span class="result-number">${index + 1}</span>
                    <strong>${movie.title}</strong>
                `;
                resultList.appendChild(li);
            });
            
            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>
"""

def load_data():
    """データの読み込みと前処理"""
    global df_movies, df_piv, rec_model
    
    # 映画データの読み込み
    df_movies = pd.read_csv("./movies_100k.csv", sep="|", encoding='latin-1')
    df_movies = df_movies[['movie_id', 'movie_title']]
    
    # 評価データの読み込み
    df = pd.read_csv("./ratings_100k.csv", sep=",")
    df = df.iloc[:, 0:3]
    df.columns = ['userId', 'movieId', 'rating']
    
    # ピボットテーブルの作成
    df_piv = df.pivot(index="movieId", columns="userId", values="rating").fillna(0)
    
    # 疎行列に変換
    df_sp = csr_matrix(df_piv.values)
    
    # モデルの学習
    rec = NearestNeighbors(n_neighbors=11, algorithm="brute", metric="cosine")
    rec_model = rec.fit(df_sp)
    
    print("データの読み込みと学習が完了しました")

def recommend_movie_by_id(movie_id, n_recommendations=5):
    """指定された映画IDから類似映画を推薦"""
    try:
        target_index = df_movies[df_movies['movie_id'] == movie_id].index[0]
        distance, indice = rec_model.kneighbors(
            df_piv.iloc[df_piv.index == target_index + 1].values.reshape(1, -1),
            n_neighbors=n_recommendations + 1,
        )
        
        recommendations = []
        for i in range(1, n_recommendations + 1):
            idx = indice.flatten()[i]
            movie_title = df_movies.loc[df_piv.index[idx], 'movie_title']
            recommendations.append(movie_title)
        
        return recommendations
    except IndexError:
        return []

def get_top_rated_movies(n=5):
    """評価の高い映画を取得（未選択時用）"""
    df_ratings = pd.read_csv("./ratings_100k.csv", sep=",")
    avg_ratings = df_ratings.groupby('movieId')['rating'].agg(['mean', 'count'])
    # 評価数が50以上の映画のみ対象
    popular_movies = avg_ratings[avg_ratings['count'] >= 50].sort_values('mean', ascending=False)
    
    top_movie_ids = popular_movies.head(n).index.tolist()
    top_movies = []
    
    for movie_id in top_movie_ids:
        movie_info = df_movies[df_movies['movie_id'] == movie_id]
        if not movie_info.empty:
            top_movies.append(movie_info.iloc[0]['movie_title'])
    
    return top_movies

@app.route('/')
def index():
    """メイン画面"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/movies')
def get_movies():
    """映画リストを返すAPI"""
    # 最初の100件を返す（表示用）
    movies_list = df_movies.head(100).to_dict('records')
    return jsonify([{'id': m['movie_id'], 'title': m['movie_title']} for m in movies_list])

@app.route('/recommend', methods=['POST'])
def recommend():
    """レコメンデーションAPI"""
    data = request.json
    movie_ids = data.get('movie_ids', [])
    
    if len(movie_ids) < 3:
        # 未選択または3つ未満の場合は総合評価の高い映画を返す
        recommendations = get_top_rated_movies(5)
    else:
        # 選択された映画から推薦を生成
        all_recommendations = []
        for movie_id in movie_ids:
            recs = recommend_movie_by_id(movie_id, n_recommendations=10)
            all_recommendations.extend(recs)
        
        # 重複を除いて上位5件を返す
        unique_recs = []
        for rec in all_recommendations:
            if rec not in unique_recs:
                unique_recs.append(rec)
            if len(unique_recs) == 5:
                break
        
        recommendations = unique_recs
    
    return jsonify({
        'recommendations': [{'title': title} for title in recommendations]
    })

if __name__ == '__main__':
    # データの読み込み
    load_data()
    # サーバー起動
    app.run(debug=True, host='0.0.0.0', port=5000)