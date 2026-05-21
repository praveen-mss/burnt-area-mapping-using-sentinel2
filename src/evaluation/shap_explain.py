import shap
import matplotlib.pyplot as plt


def compute_shap_values(model, X_sample, feature_names, save_path=None):

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=feature_names,
        show=False
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

    return shap_values