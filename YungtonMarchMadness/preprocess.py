import csv
import random
import re

def loaddata(filename):
    data = []
    with open(filename, 'rU') as csvfile:
        reader = csv.reader(csvfile)
        # Skip header lines
        reader.next()
        for row in reader:
           data.append(row)
    return data


def create_season_files():
    reg_season = loaddata('data/RegularSeasonStats.csv')
    year = reg_season[0][len(reg_season[0]) - 1]
    rows = []
    for row in reg_season:
        if year != row[len(row) - 1]:
            #write rows to file
            with open('data/RegSeasonStats' + year + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            year = row[len(row) - 1]
            rows = []
        else:
            rows.append(row)
    with open('data/RegSeasonStats' + year + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)


def make_features(year):
    LEN_FEATS = 13
    reg_season = loaddata('data/RegSeasonStats' + year + '.csv')
    # seeds = loaddata('data/TourneySeeds.csv')

    for i in range(len(reg_season)):
        reg_season[i] = map(float, reg_season[i])
    stats = {}
    game_count = {}

    for row in reg_season:
        team1 = row[0]
        team2 = row[1]
        if not stats.has_key(team1):
            stats[team1] = [0.0] * LEN_FEATS
        if not stats.has_key(team2):
            stats[team2] = [0.0] * LEN_FEATS

        if not game_count.has_key(team1):
            game_count[team1] = 1
        else:
            game_count[team1] += 1

        if not game_count.has_key(team2):
            game_count[team2] = 1
        else:
            game_count[team2] += 1

        for i in range(2, 1 + (2 * LEN_FEATS)):
            if i <= 14:
                stats[team1][i-2] += row[i]
            else:
                stats[team2][i-15] += row[i]

    #Get averages
    for team in stats:
        stats[team] = [x / game_count[team] for x in stats[team]]

    # for row in seeds:
    #     if row[0] == year:
    #         team = float(row[2])
    #         seed = int(re.findall('\d+', row[1])[0])
    #         stats[team].append(seed)

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

    with open('data/avg_team_stats' + year + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for team in stats:
            writer.writerow([team] + stats[team])

    with open('data/training_data' + year + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train)

def make_test_data(year):
    tourney_data = loaddata('data/TourneyDetailedResults.csv')
    stats = loaddata('data/avg_team_stats' + year + '.csv')
    team_stats = {}
    for row in stats:
        team_stats[row[0][:-2]] = row[1:]
    test = []
    for row in tourney_data:
        if int(row[0]) == int(year):
            if random.randint(0, 1) == 0:
                test_row = []
                team1 = row[2]
                team2 = row[4]
                test_row.extend(team_stats[team1])
                test_row.extend(team_stats[team2])
                test_row.append(1)
                test.append(test_row)
            else:
                test_row = []
                team1 = row[4]
                team2 = row[2]
                test_row.extend(team_stats[team1])
                test_row.extend(team_stats[team2])
                test_row.append(0)
                test.append(test_row)
    with open('data/test_data' + year + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test)
    return test

def make_test_data_w_years_teams(year):
    tourney_data = loaddata('data/TourneyDetailedResults.csv')
    stats = loaddata('data/avg_team_stats' + year + '.csv')
    team_stats = {}
    for row in stats:
        # because the team number is a float
        team_stats[row[0][:-2]] = row[1:]
    test = []
    for row in tourney_data:
        if int(row[0]) == int(year):
            test_row = []
            if row[2] < row[4]:
                team1 = row[2]
                team2 = row[4]
                test_row.append(team1)
                test_row.append(team2)
                test_row.append(int(row[0]))
                test_row.extend(team_stats[team1])
                test_row.extend(team_stats[team2])
                test_row.append(1)
                test.append(test_row)
            else:
                team1 = row[4]
                team2 = row[2]
                test_row.append(team1)
                test_row.append(team2)
                test_row.append(int(row[0]))
                test_row.extend(team_stats[team1])
                test_row.extend(team_stats[team2])
                test_row.append(0)
                test.append(test_row)

    with open('data/test_data_w_years_teams' + year + '.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test)
    return test


def make_final_train():
    LEN_FEATS = 13
    reg_season = loaddata('data/2017_Final_DetailedResults.csv')
    reg_season.pop(0)
    for i in range(len(reg_season)):
        reg_season[i] = map(float, reg_season[i])
    stats = {}
    game_count = {}

    for row in reg_season:
        team1 = row[0]
        team2 = row[1]
        if not stats.has_key(team1):
            stats[team1] = [0.0] * LEN_FEATS
        if not stats.has_key(team2):
            stats[team2] = [0.0] * LEN_FEATS

        if not game_count.has_key(team1):
            game_count[team1] = 1
        else:
            game_count[team1] += 1

        if not game_count.has_key(team2):
            game_count[team2] = 1
        else:
            game_count[team2] += 1

        for i in range(2, 1 + (2 * LEN_FEATS)):
            if i <= 14:
                stats[team1][i-2] += row[i]
            else:
                stats[team2][i-15] += row[i]

    #Get averages
    for team in stats:
        stats[team] = [x / game_count[team] for x in stats[team]]

    # for row in seeds:
    #     if row[0] == year:
    #         team = float(row[2])
    #         seed = int(re.findall('\d+', row[1])[0])
    #         stats[team].append(seed)

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

    with open('data/avg_team_stats2017.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        for team in stats:
            writer.writerow([team] + stats[team])

    with open('data/training_data2017.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(train)
    return train


def make_final_test():
    tourney = loaddata('data/SampleSubmission_5050Benchmark.csv')
    tourney.pop(0)
    stats = loaddata('data/avg_team_stats2017.csv')
    team_stats = {}
    for row in stats:
        # because the team number is a float
        team_stats[row[0][:-2]] = row[1:]
    test = []
    for row in tourney:
        test_row = []
        parts = row[0].split('_')
        team1 = parts[1]
        team2 = parts[2]
        test_row.append(team1)
        test_row.append(team2)
        test_row.extend(team_stats[team1])
        test_row.extend(team_stats[team2])
        test_row.append(None)
        test.append(test_row)
    with open('data/final_test_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(test)
    return test


def merge_hist_data(years):
    hist_data = []
    for year in years:
        this_hist = loaddata('data/historical_csv' + year + '.csv')
        hist_data.extend(this_hist)
    with open('data/historical_data.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(hist_data)
