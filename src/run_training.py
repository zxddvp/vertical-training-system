from src import data_processing, scheduler, train, evaluate


def main():
    scheduler.init_cluster()
    print('Resources:', scheduler.get_cluster_resources())
    df = data_processing.load_dataset('data/financial.csv')
    parts = data_processing.vertical_split(df, [['age', 'income'], ['credit_score']], 'label')
    model, merged = train.train(parts, 'label', epochs=1)
    acc = evaluate.accuracy(model, merged, 'label')
    print('Accuracy:', acc)


if __name__ == '__main__':
    main()