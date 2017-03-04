import csv
import random

def loaddata(filename):
    data = []
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header lines
        reader.next()
        for row in reader:
           data.append(row)
    return data


def make_features():
    reg_season = loaddata('data/RegularSeasonStats.csv')
    for i in range(len(reg_season)):
        reg_season[i] = map(float, reg_season[i])
    stats = {}
    game_count = {}

    for row in reg_season:
        #Possibly randomly decide whether to put team 1 or 2 first so result isn't always 1
        team1 = row[0]
        team2 = row[1]
        if not stats.has_key(team1):
            stats[team1] = [0.0] * 13
        if not stats.has_key(team2):
            stats[team2] = [0.0] * 13

        if not game_count.has_key(team1):
            game_count[team1] = 1
        else:
            game_count[team1] += 1

        if not game_count.has_key(team2):
            game_count[team2] = 1
        else:
            game_count[team2] += 1

        for i in range(2, len(row) - 1):
            if i <= 14:
                stats[team1][i-2] += row[i]
            else:
                stats[team2][i-15] += row[i]

    #Get averages
    for team in stats:
        stats[team] = [x / game_count[team] for x in stats[team]]

    train = []
    for row in reg_season:
        # get two teams, append their averages, get who wins
        if random.randint(0,1) == 0:
            train_row = []
            team1 = row[0]
            team2 = row[1]
            train_row.extend(stats[team1])
            train_row.extend(stats[team2])
            train_row.append(1)
            train.append(train_row)
        else:
            train_row = []
            team1 = row[1]
            team2 = row[0]
            train_row.extend(stats[team1])
            train_row.extend(stats[team2])
            train_row.append(0)
            train.append(train_row)

    with open('data/avg_team_stats.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for team in stats:
            writer.writerow([team] + stats[team])

    with open('data/training_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train)

