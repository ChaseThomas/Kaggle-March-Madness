import combine_csvs
import model

def main():
    combined = combine_csvs.combine_csvs({
        "train": ["season", "team1", "team2", "actual"],
        "test": ["team1", "team2"],
    })
    train = combined["train"]
    test = combined["test"]
    model.run_ensemble(train, test)

main()
