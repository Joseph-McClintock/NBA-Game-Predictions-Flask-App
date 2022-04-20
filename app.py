from flask import Flask, render_template, request
import matplotlib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href="")
    else:
        seasons = request.form['seasons']
        print(seasons)
        calculate_model(seasons)
        piechart = "static/images/piechart.png"
        scatterplot = "static/images/scatterplot.png"
        barchart = "static/images/barchart.png"
        return render_template('index.html', href1=piechart, href2=scatterplot, href3=barchart)


def calculate_model(picked_season):

    df_game_data = pd.read_csv('static/csvfiles/traindata' + picked_season + ".csv", sep='\t', encoding='utf-8', index_col=0)
    df_all_teams = pd.read_csv('static/csvfiles/teamelo' + picked_season + ".csv", sep='\t', encoding='utf-8', index_col=0)

    X = df_game_data.drop(columns=['GAME_ID', 'MATCHUP', 'WL'])
    y = df_game_data['WL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, test_size=.3, shuffle=False)
 
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    score = accuracy_score(y_test, predictions)
    
    # Piechart for win and loss %
    wl = ['Wins', 'Losses']
    wrong = 1.00 - score
    correct_incorrect = [score, wrong]
    plt.pie(correct_incorrect,
            labels=wl,
            explode=(0.025, 0),
            autopct='%1.1f%%')
    plt.title('Win Loss Graph ' + str(picked_season))
    plt.savefig('static/images/piechart.png')
    plt.clf()

    # Scatterplot for wins to ELO
    plt.title('Wins to ELO ' + str(picked_season))
    plt.xlabel('ELO')
    plt.ylabel('Wins')
    team_elo = df_all_teams['ELO'].tolist()
    team_wins = df_all_teams['Wins'].tolist()
    plt.scatter(team_elo,team_wins, label='Elo to Losses')
    plt.savefig('static/images/scatterplot.png')
    plt.clf()

    wl_list = [0,0]
    for result in predictions:
        if result == 'W':
            wl_list[0] += 1
        elif result == 'L':
            wl_list[1] += 1

    # Bar graph to show how many wins and losses the model guessed
    plt.title('Prediction guesses for ' + str(picked_season))
    plt.bar(wl, wl_list)
    plt.savefig('static/images/barchart.png')
    plt.clf()