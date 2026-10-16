import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

# Настройка страницы
st.set_page_config(
    page_title="Titanic ML Studio Pro",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


# 1. Загрузка исходных данных
@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


# 2. Конструирование признаков (Feature Engineering) и очистка
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # Извлечение титула из имени
    if "Name" in data.columns:
        data["Title"] = data["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
        rare_titles = [
            "Lady", "Countess", "Capt", "Col", "Don", "Dr",
            "Major", "Rev", "Sir", "Jonkheer", "Dona",
        ]
        data["Title"] = data["Title"].replace(rare_titles, "Rare")
        data["Title"] = data["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        data["Title"] = data["Title"].fillna("Mr")
    else:
        data["Title"] = "Mr"

    # Индикатор наличия известного номера каюты
    if "Cabin" in data.columns:
        data["HasCabin"] = data["Cabin"].notna().astype(int)
    else:
        data["HasCabin"] = 0

    # Заполнение пропусков возраста медианой по группе (Класс + Пол)
    data["Age"] = data.groupby(["Pclass", "Sex"])["Age"].transform(
        lambda s: s.fillna(s.median())
    )
    data["Age"] = data["Age"].fillna(data["Age"].median())

    # Заполнение порта посадки и стоимости билета
    data["Embarked"] = data["Embarked"].fillna("S")
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())

    # Признаки семьи
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    # Кодирование категориальных признаков в числа
    data["Sex_Code"] = data["Sex"].map({"male": 0, "female": 1}).fillna(0).astype(int)
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    data["Title_Code"] = data["Title"].map(title_mapping).fillna(1).astype(int)
    emb_mapping = {"S": 0, "C": 1, "Q": 2}
    data["Embarked_Code"] = data["Embarked"].map(emb_mapping).fillna(0).astype(int)

    return data


FEATURE_COLUMNS = [
    "Pclass",
    "Sex_Code",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "Title_Code",
    "Embarked_Code",
]

FEATURE_LABELS = {
    "Pclass": "Класс билета (Pclass)",
    "Sex_Code": "Пол (Sex)",
    "Age": "Возраст (Age)",
    "SibSp": "Братья/сестры/супруги (SibSp)",
    "Parch": "Родители/дети (Parch)",
    "Fare": "Цена билета (Fare)",
    "FamilySize": "Размер семьи (FamilySize)",
    "IsAlone": "Путешествует один (IsAlone)",
    "HasCabin": "Известна каюта (HasCabin)",
    "Title_Code": "Социальный титул (Title)",
    "Embarked_Code": "Порт посадки (Embarked)",
}


# 3. Обучение активной модели + сравнение всех 3 алгоритмов
@st.cache_resource
def train_and_compare_models(df: pd.DataFrame, active_model_name: str, n_estimators: int, max_depth: int):
    processed = preprocess_dataframe(df)
    X = processed[FEATURE_COLUMNS]
    y = processed["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models_dict = {
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=min(max_depth, 6), random_state=42
        ),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }

    comparison_rows = []
    roc_curves = {}
    trained_models = {}
    cms = {}
    importances = {}

    for name, clf in models_dict.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        comparison_rows.append({
            "Алгоритм": name,
            "Accuracy": round(acc, 3),
            "Precision": round(prec, 3),
            "Recall": round(rec, 3),
            "F1-Score": round(f1, 3),
            "ROC-AUC": round(auc, 3),
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_curves[name] = (fpr, tpr, auc)
        trained_models[name] = clf
        cms[name] = confusion_matrix(y_test, y_pred)

        if hasattr(clf, "feature_importances_"):
            raw_imp = clf.feature_importances_
        else:
            raw_imp = np.abs(clf.coef_[0])
            raw_imp = raw_imp / raw_imp.sum()

        importances[name] = pd.DataFrame({
            "Признак": [FEATURE_LABELS[col] for col in FEATURE_COLUMNS],
            "Важность": raw_imp,
        }).sort_values(by="Важность", ascending=True)

    comparison_df = pd.DataFrame(comparison_rows)
    active_model = trained_models[active_model_name]
    active_cm = cms[active_model_name]
    active_imp = importances[active_model_name]
    active_metrics = comparison_df[comparison_df["Алгоритм"] == active_model_name].iloc[0].to_dict()

    return active_model, active_metrics, active_cm, active_imp, comparison_df, roc_curves, processed


# 4. Функция расчета индивидуального вклада признаков для конкретного пассажира
def explain_passenger_prediction(model, df_processed: pd.DataFrame, user_row: pd.DataFrame) -> pd.DataFrame:
    baseline_row = df_processed[FEATURE_COLUMNS].median().to_frame().T
    base_prob = float(model.predict_proba(baseline_row)[0][1])

    feature_groups = {
        "Пол и титул": ["Sex_Code", "Title_Code"],
        "Класс билета": ["Pclass"],
        "Возраст": ["Age"],
        "Размер семьи": ["SibSp", "Parch", "FamilySize", "IsAlone"],
        "Цена билета": ["Fare"],
        "Наличие каюты": ["HasCabin"],
        "Порт посадки": ["Embarked_Code"],
    }

    impacts = []
    for label, cols in feature_groups.items():
        temp_row = user_row.copy()
        for c in cols:
            temp_row[c] = baseline_row[c].values[0]
        prob_without = float(model.predict_proba(temp_row)[0][1])
        user_prob = float(model.predict_proba(user_row)[0][1])
        diff_pct = (user_prob - prob_without) * 100.0
        impacts.append({
            "Фактор": label,
            "Вклад (%)": round(diff_pct, 1),
            "Тип влияния": "Повышает шанс" if diff_pct >= 0 else "Снижает шанс",
        })

    return pd.DataFrame(impacts).sort_values(by="Вклад (%)", ascending=True)


# Загрузка данных
df_raw = load_raw_data()

# Боковая панель настроек модели
st.sidebar.title("⚙️ Настройки ML-модели")
selected_model = st.sidebar.selectbox(
    "Активный алгоритм классификации",
    ["Random Forest", "Gradient Boosting", "Logistic Regression"],
)

if selected_model in ["Random Forest", "Gradient Boosting"]:
    n_trees = st.sidebar.slider("Количество деревьев (n_estimators)", 50, 300, 150, step=25)
    tree_depth = st.sidebar.slider("Максимальная глубина (max_depth)", 3, 12, 6)
else:
    n_trees, tree_depth = 150, 6

model, metrics, cm, importance_df, comparison_df, roc_curves, df_processed = train_and_compare_models(
    df_raw, selected_model, n_trees, tree_depth
)

# Заголовок и верхняя панель KPI
st.title("🚢 Titanic ML Studio: Анализ данных и прогноз выживаемости")
st.markdown(
    "Полноценное аналитическое приложение: разведочный анализ (EDA), конструирование признаков, "
    "сравнение алгоритмов машинного обучения и индивидуальный либо пакетный прогноз."
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Пассажиров в базе", f"{len(df_raw)}")
kpi2.metric("Базовая выживаемость", f"{df_raw['Survived'].mean() * 100:.1f}%")
kpi3.metric(f"Точность ({selected_model})", f"{metrics['Accuracy'] * 100:.1f}%")
kpi4.metric("ROC-AUC модели", f"{metrics['ROC-AUC']:.3f}")

# Вкладки приложения
tab_pred, tab_eda, tab_ml, tab_batch = st.tabs([
    "🎯 Калькулятор прогноза",
    "📊 Разведочный анализ (EDA)",
    "🧠 Метрики и сравнение моделей",
    "📁 Пакетный прогноз (CSV / Kaggle)",
])

# --- ВКЛАДКА 1: ИНДИВИДУАЛЬНЫЙ ПРОГНОЗ ---
with tab_pred:
    st.subheader("Индивидуальный расчет вероятности спасения")
    col_form, col_result = st.columns([1.15, 1])

    with col_form:
        c1, c2 = st.columns(2)
        with c1:
            pclass_in = st.selectbox("Класс билета (Pclass)", [1, 2, 3], index=0)
            sex_in = st.radio("Пол пассажира", ["Женский", "Мужской"], horizontal=True)

            # Умная фильтрация титулов в зависимости от выбранного пола
            if sex_in == "Женский":
                title_options = [
                    "Miss (Девушка / не замужем)",
                    "Mrs (Замужняя женщина)",
                    "Rare (Леди / Графиня / Доктор)",
                ]
            else:
                title_options = [
                    "Mr (Взрослый мужчина)",
                    "Master (Мальчик / ребенок)",
                    "Rare (Капитан / Полковник / Доктор / Сэр)",
                ]

            title_in = st.selectbox("Титул обращения", title_options)
            age_in = st.slider("Возраст (лет)", 1, 80, 26)

        with c2:
            sibsp_in = st.number_input("Братьев/сестер или супругов на борту (SibSp)", 0, 8, 0)
            parch_in = st.number_input("Родителей или детей на борту (Parch)", 0, 6, 0)
            fare_in = st.slider("Стоимость билета (£)", 5.0, 260.0, 35.0, step=1.0)
            emb_in = st.selectbox("Порт посадки", ["C — Шербур", "S — Саутгемптон", "Q — Квинстаун"])
            cabin_in = st.checkbox("Номер каюты известен (палуба зафиксирована)", value=True)

    # Преобразование введенных параметров в вектор признаков
    sex_code = 1 if sex_in == "Женский" else 0
    title_key = title_in.split(" ")[0]
    title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    title_code = title_map.get(title_key, 1)
    emb_key = emb_in.split(" ")[0]
    emb_map = {"S": 0, "C": 1, "Q": 2}
    emb_code = emb_map.get(emb_key, 0)
    family_size = sibsp_in + parch_in + 1
    is_alone = 1 if family_size == 1 else 0
    has_cabin = 1 if cabin_in else 0

    user_row = pd.DataFrame(
        [[pclass_in, sex_code, age_in, sibsp_in, parch_in, fare_in, family_size, is_alone, has_cabin, title_code, emb_code]],
        columns=FEATURE_COLUMNS,
    )

    surv_prob = float(model.predict_proba(user_row)[0][1])

    with col_result:
        st.markdown("### Вердикт модели")
        if surv_prob >= 0.5:
            st.success(f"### Пассажир ВЫЖИВЕТ\n**Вероятность спасения: {surv_prob * 100:.1f}%**")
        else:
            st.error(f"### Пассажир ПОГИБНЕТ\n**Вероятность спасения: {surv_prob * 100:.1f}%**")

        st.progress(surv_prob)
        st.write(
            f"**Размер семьи:** {family_size} чел. | "
            f"**Средний шанс в {pclass_in}-м классе:** {df_raw[df_raw['Pclass'] == pclass_in]['Survived'].mean() * 100:.1f}% | "
            f"**Отклонение от среднего (38.4%):** {(surv_prob - 0.384) * 100:+.1f}%"
        )

        # График вклада факторов для этого пассажира
        impact_df = explain_passenger_prediction(model, df_processed, user_row)
        fig_impact = px.bar(
            impact_df,
            x="Вклад (%)",
            y="Фактор",
            color="Тип влияния",
            orientation="h",
            text="Вклад (%)",
            color_discrete_map={"Повышает шанс": "#2ecc71", "Снижает шанс": "#e74c3c"},
            title="Как параметры этого пассажира повлияли на прогноз (%)",
        )
        fig_impact.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), showlegend=True)
        st.plotly_chart(fig_impact, use_container_width=True)

# --- ВКЛАДКА 2: РАЗВЕДОЧНЫЙ АНАЛИЗ (EDA) ---
with tab_eda:
    st.subheader("Исследование закономерностей в датасете")
    g1, g2 = st.columns(2)

    with g1:
        surv_by_class_sex = (
            df_processed.groupby(["Pclass", "Sex"], as_index=False)["Survived"]
            .mean()
            .assign(Survived_Pct=lambda d: (d["Survived"] * 100).round(1))
        )
        fig_cs = px.bar(
            surv_by_class_sex,
            x="Pclass",
            y="Survived_Pct",
            color="Sex",
            barmode="group",
            text="Survived_Pct",
            title="Выживаемость по классу каюты и полу (%)",
            labels={"Pclass": "Класс билета", "Survived_Pct": "Выжили (%)", "Sex": "Пол"},
        )
        st.plotly_chart(fig_cs, use_container_width=True)

    with g2:
        surv_by_title = (
            df_processed.groupby("Title", as_index=False)["Survived"]
            .agg(Всего="count", Выживаемость="mean")
            .assign(Выживаемость=lambda d: (d["Выживаемость"] * 100).round(1))
        )
        fig_title = px.bar(
            surv_by_title,
            x="Title",
            y="Выживаемость",
            color="Title",
            text="Выживаемость",
            title="Выживаемость по извлеченному титулу (%)",
            labels={"Title": "Титул", "Выживаемость": "Выжили (%)"},
        )
        st.plotly_chart(fig_title, use_container_width=True)

    fig_age = px.histogram(
        df_processed,
        x="Age",
        color=df_processed["Survived"].map({1: "Выжил", 0: "Погиб"}),
        nbins=30,
        barmode="overlay",
        title="Распределение возраста выживших и погибших пассажиров",
        labels={"Age": "Возраст (лет)", "color": "Исход"},
    )
    st.plotly_chart(fig_age, use_container_width=True)

    with st.expander("Показать таблицу обработанного датасета (с новыми признаками)"):
        st.dataframe(df_processed.head(50), use_container_width=True)

# --- ВКЛАДКА 3: МЕТРИКИ И СРАВНЕНИЕ МОДЕЛЕЙ ---
with tab_ml:
    st.subheader(f"Диагностика активной модели: {selected_model}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['Precision']:.3f}")
    m3.metric("Recall", f"{metrics['Recall']:.3f}")
    m4.metric("F1-Score", f"{metrics['F1-Score']:.3f}")
    m5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")

    col_imp, col_cm = st.columns([1.3, 1])

    with col_imp:
        fig_imp = px.bar(
            importance_df,
            x="Важность",
            y="Признак",
            orientation="h",
            title=f"Вклад признаков в решение модели ({selected_model})",
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_cm:
        cm_df = pd.DataFrame(
            cm,
            index=["Факт: Погиб (0)", "Факт: Выжил (1)"],
            columns=["Прогноз: Погиб (0)", "Прогноз: Выжил (1)"],
        )
        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            title=f"Матрица ошибок ({selected_model})",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()
    st.subheader("Сводное сравнение всех 3 алгоритмов на тестовой выборке")
    col_table, col_roc = st.columns([1.1, 1])

    with col_table:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        best_algo = comparison_df.sort_values(by="ROC-AUC", ascending=False).iloc[0]
        st.info(
            f"🏆 Лучшая разделяющая способность по ROC-AUC: **{best_algo['Алгоритм']}** "
            f"(ROC-AUC = **{best_algo['ROC-AUC']:.3f}**, Accuracy = **{best_algo['Accuracy'] * 100:.1f}%**)."
        )

    with col_roc:
        fig_roc = go.Figure()
        for algo_name, (fpr, tpr, auc_val) in roc_curves.items():
            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{algo_name} (AUC = {auc_val:.3f})",
                )
            )
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Случайное угадывание (AUC = 0.500)",
                line=dict(dash="dash", color="gray"),
            )
        )
        fig_roc.update_layout(
            title="Сравнение ROC-кривых трех моделей",
            xaxis_title="Доля ложных тревог (False Positive Rate)",
            yaxis_title="Полнота обнаружения выживших (True Positive Rate)",
            height=380,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

# --- ВКЛАДКА 4: ПАКЕТНЫЙ ПРОГНОЗ (CSV + KAGGLE) ---
with tab_batch:
    st.subheader("Пакетное прогнозирование и генерация файла для Kaggle")
    mode = st.radio(
        "Выбери источник данных для пакетного расчета:",
        ["🎲 Использовать встроенную демо-выборку (25 пассажиров)", "📂 Загрузить собственный CSV-файл"],
        horizontal=True,
    )

    batch_df = None
    if mode.startswith("🎲"):
        batch_df = df_raw.sample(25, random_state=42).drop(columns=["Survived"], errors="ignore")
    else:
        uploaded_file = st.file_uploader("Загрузи CSV-файл (например, test.csv с Kaggle)", type=["csv"])
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)

    if batch_df is not None:
        proc_batch = preprocess_dataframe(batch_df)
        preds = model.predict(proc_batch[FEATURE_COLUMNS])
        probs = model.predict_proba(proc_batch[FEATURE_COLUMNS])[:, 1]

        result_df = batch_df.copy()
        result_df["Predicted_Survived"] = preds
        result_df["Survival_Probability"] = probs.round(3)

        st.success(
            f"Обработано пассажиров: **{len(result_df)}**. "
            f"По прогнозу выживут: **{int(preds.sum())} чел.** ({preds.mean() * 100:.1f}%)."
        )
        st.dataframe(result_df.head(30), use_container_width=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            full_csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Скачать полный отчет с вероятностями (CSV)",
                data=full_csv,
                file_name="titanic_predictions_full.csv",
                mime="text/csv",
            )
        with dl2:
            if "PassengerId" in result_df.columns:
                kaggle_df = pd.DataFrame({
                    "PassengerId": result_df["PassengerId"],
                    "Survived": preds,
                })
            else:
                kaggle_df = pd.DataFrame({
                    "PassengerId": np.arange(892, 892 + len(preds)),
                    "Survived": preds,
                })
            kaggle_csv = kaggle_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="🏆 Скачать готовый сабмит для Kaggle (PassengerId, Survived)",
                data=kaggle_csv,
                file_name="kaggle_submission.csv",
                mime="text/csv",
            )
