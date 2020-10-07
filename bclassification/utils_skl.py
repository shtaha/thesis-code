import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

from lib.visualizer import pprint


def describe_dataset(x, y, name):
    labels = np.unique(y)
    count = [np.count_nonzero(y == c) for c in labels]
    weights = [c / sum(count) for c in count]

    pprint(f"    - {name}", "X" + str(x.shape), "Y" + str(y.shape))
    pprint("        - Labels:", str(labels))
    pprint("        - Count:", str(count))
    pprint("        - Weights:", str(weights))
    return labels, count, weights


def train(
    train_data,
    test_data,
    scaling=False,
    power_scaling=False,
    random_search=(False,),
    cross_validation=(False, 5, "balanced_accuracy"),
):
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Load data
    describe_dataset(x_train, y_train, "Train")
    describe_dataset(x_test, y_test, "Test")

    # Feature scaling
    if scaling:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    if power_scaling:
        ptransformer = PowerTransformer()
        ptransformer.fit(x_train)
        x_train = ptransformer.transform(x_train)
        x_test = ptransformer.transform(x_test)

    # Training
    best_est = dict()
    if random_search[0]:
        for key in random_search[1]:
            cls = random_search[1][key]["cls"]
            rs = RandomizedSearchCV(
                cls,
                param_distributions=random_search[1][key]["param_dist"],
                n_iter=random_search[2],
                cv=5,
                n_jobs=-1,
                random_state=0,
                scoring="f1_micro",
            )
            rs.fit(x_train, y_train.ravel())
            best_est[key] = rs.best_estimator_
            pprint(f"    - {key}", str(rs.best_estimator_))

    if not random_search[0]:
        est = random_search[3]
    else:
        est = []
        for key in best_est:
            est.append((key, best_est[key]))

    vc = VotingClassifier(
        estimators=est, voting="soft", weights=[1 for _ in range(len(est))], n_jobs=-1,
    )
    vc.fit(x_train, y_train.ravel())

    y_train_pred = vc.predict(x_train)
    y_test_pred = vc.predict(x_test)

    pprint("    - Accuracy:", "")
    pprint("        - Train:", accuracy_score(y_train, y_train_pred))
    pprint("        - Test:", accuracy_score(y_test, y_test_pred))

    pprint("    - MCC:", "")
    pprint("        - Train:", matthews_corrcoef(y_train, y_train_pred))
    pprint("        - Test:", matthews_corrcoef(y_test, y_test_pred))

    p_train, r_train, f_train, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average="binary"
    )
    p_test, r_test, f_test, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary"
    )

    pprint("    - Precision:", "")
    pprint("        - Train:", p_train)
    pprint("        - Test:", p_test)

    pprint("    - Recall:", "")
    pprint("        - Train:", r_train)
    pprint("        - Test:", r_test)

    pprint("    - F1-Score:", "")
    pprint("        - Train:", f_train)
    pprint("        - Test:", f_test)

    if cross_validation[0]:
        pprint("    - CV:", f"{cross_validation[1]}-fold")
        score = cross_val_score(
            vc,
            x_train,
            y_train.ravel(),
            cv=cross_validation[1],
            scoring=cross_validation[2],
            n_jobs=-1,
        )
        pprint("        - Mean:", score.mean())
        pprint("        - Std:", score.std())


def train_model(
    train_data,
    test_data,
    model,
    scaling=True,
    power_scaling=False,
    random_search=(False,),
    cross_validation=(False, 5, "balanced_accuracy"),
):
    x_train, y_train = train_data
    x_test, y_test = test_data

    # Load data
    describe_dataset(x_train, y_train, "Train")
    describe_dataset(x_test, y_test, "Test")

    # Feature scaling
    if scaling:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)

    if power_scaling:
        ptransformer = PowerTransformer()
        ptransformer.fit(x_train)
        x_train = ptransformer.transform(x_train)
        x_test = ptransformer.transform(x_test)

    # Training
    best_est = None
    if random_search[0]:
        rs = RandomizedSearchCV(
            model,
            param_distributions=random_search[1],
            n_iter=random_search[2],
            cv=5,
            n_jobs=-1,
            random_state=0,
            scoring=random_search[3],
        )
        rs.fit(x_train, y_train.ravel())
        best_est = rs.best_estimator_
        pprint(f"    - Random Search best:", str(rs.best_estimator_))

    if best_est:
        model = best_est

    model.fit(x_train, y_train.ravel())

    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    pprint("    - Accuracy:", "")
    pprint("        - Train:", accuracy_score(y_train, y_train_pred))
    pprint("        - Test:", accuracy_score(y_test, y_test_pred))

    pprint("    - MCC:", "")
    pprint("        - Train:", matthews_corrcoef(y_train, y_train_pred))
    pprint("        - Test:", matthews_corrcoef(y_test, y_test_pred))

    p_train, r_train, f_train, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average="binary"
    )
    p_test, r_test, f_test, _ = precision_recall_fscore_support(
        y_test, y_test_pred, average="binary"
    )

    pprint("    - Precision:", "")
    pprint("        - Train:", p_train)
    pprint("        - Test:", p_test)

    pprint("    - Recall:", "")
    pprint("        - Train:", r_train)
    pprint("        - Test:", r_test)

    pprint("    - F1-Score:", "")
    pprint("        - Train:", f_train)
    pprint("        - Test:", f_test)

    if cross_validation[0]:
        pprint("    - CV:", f"{cross_validation[1]}-fold")
        score = cross_val_score(
            model,
            np.vstack((x_train, x_test)),
            np.vstack((y_train, y_test)).ravel(),
            cv=cross_validation[1],
            scoring=cross_validation[2],
            n_jobs=-1,
        )
        pprint("        - Mean:", score.mean())
        pprint("        - Std:", score.std())
