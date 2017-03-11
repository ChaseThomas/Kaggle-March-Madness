import csv

def standardize():
    new_rows = []

    with open("submission.csv", "r") as submission:
        reader = csv.reader(submission)
        for row in reader:
            if len(row[0].split("_")) > 1:
                split_key = [int(i) for i in row[0].split("_")]
                new_row = split_key + row[1:]
                new_rows.append(new_row)

    with open("new_submission.csv", "w") as outfile:
        writer = csv.writer(outfile)
        for row in new_rows:
            writer.writerow(row)

standardize()
