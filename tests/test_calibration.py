from __future__ import annotations

import pytest

import scripts.calibrate_threshold as calib


# ----------------------------------------------------------------------
# 10. El calibrador pondera correctamente los costes.
# ----------------------------------------------------------------------
def test_evaluate_threshold_matriz_de_confusion_correcta():
    answerable = [0.9, 0.8, 0.5]
    unanswerable = [0.6, 0.2]
    metrics = calib.evaluate_threshold(
        0.55, answerable, unanswerable, cost_fp=1.0, cost_fn=1.0
    )
    # umbral 0.55: respondibles >=0.55 -> 0.9 y 0.8 responden bien (TP=2),
    # 0.5 queda por debajo -> abstiene debiendo responder (FN=1).
    # no respondibles: 0.6 >= 0.55 -> responde debiendo abstenerse (FP=1),
    # 0.2 < 0.55 -> abstiene correctamente (TN=1).
    assert metrics.true_positive == 2
    assert metrics.false_negative == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.cost == pytest.approx(2.0)


def test_coste_alto_en_falsos_positivos_prioriza_no_responder_mal():
    # Distribuciones solapadas a propósito: 0.5 es respondible con score
    # bajo y 0.45 es no respondible con score alto. Ningún umbral acierta
    # en los dos a la vez.
    answerable = [0.9, 0.8, 0.5]
    unanswerable = [0.45, 0.2, 0.1]

    costoso_responder_mal = calib.calibrate(
        answerable, unanswerable, cost_fp=10.0, cost_fn=1.0
    )
    costoso_abstenerse_de_mas = calib.calibrate(
        answerable, unanswerable, cost_fp=1.0, cost_fn=10.0
    )

    # Con FP caro, el umbral óptimo no debe dejar pasar el 0.45 no
    # respondible: su recomendación debe quedar por encima de 0.45.
    assert costoso_responder_mal["mejor"]["responde_debiendo_abstenerse_FP"] == 0
    # Con FN caro, el sistema prefiere responder aunque a veces se
    # equivoque con una no respondible: tolera algún FP a cambio de casi
    # ningún FN.
    assert (
        costoso_abstenerse_de_mas["mejor"]["abstiene_debiendo_responder_FN"]
        <= costoso_responder_mal["mejor"]["abstiene_debiendo_responder_FN"]
    )


def test_calibrate_sin_casos_de_un_tipo_avisa():
    result = calib.calibrate([], [0.1, 0.2], cost_fp=1.0, cost_fn=1.0)
    assert "faltan preguntas etiquetadas" in result["veredicto"]


def test_calibrate_separacion_amplia_sin_solape():
    result = calib.calibrate([0.8, 0.9], [0.1, 0.2], cost_fp=3.0, cost_fn=1.0)
    assert result["margen"] > 0
    assert "amplia" in result["veredicto"]
    assert result["mejor"]["responde_debiendo_abstenerse_FP"] == 0
    assert result["mejor"]["abstiene_debiendo_responder_FN"] == 0


def test_calibrate_expone_tabla_cobertura_vs_exactitud():
    result = calib.calibrate([0.6, 0.7], [0.2, 0.3], cost_fp=3.0, cost_fn=1.0)
    assert result["tabla"]
    row = result["tabla"][0]
    assert "cobertura" in row
    assert "exactitud_si_responde" in row
    assert "balanced_accuracy" in row
