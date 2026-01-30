import optuna
import matplotlib.pyplot as plt

x_values = []


def test_objective(trial):
    x = trial.suggest_float("x", 0.8, 1, log=True)
    x_values.append(x)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(test_objective, n_trials=1000)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Plot the objective function landscape
    x_values.sort()
    plt.figure()
    plt.scatter(range(len(x_values)), x_values)
    plt.xlabel("Trial")
    plt.ylabel("x value")
    plt.title("Objective Function Landscape")
    plt.show()
