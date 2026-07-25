# Model Card - axial_t2_alkafri_final_v2_candidate.pt

Proposito academico: segmentacion axial multiclass de RM lumbar para investigacion.

Arquitectura: AxialUNet2D; target size: (256, 256); mapping raw->class: {250: 0, 0: 1, 50: 2, 100: 3, 150: 4, 200: 5}.

raw0 boost: 1.0; monitor metric: dice_macro_foreground; best epoch: 17; best validation metric: 0.7283182698789201.

Validation metrics: {
  "confusionMatrix": [
    [
      4851743,
      94747,
      6192,
      9477,
      263,
      6916
    ],
    [
      6684,
      7775,
      0,
      0,
      0,
      0
    ],
    [
      4346,
      0,
      181493,
      21,
      188,
      834
    ],
    [
      4515,
      0,
      87,
      82741,
      263,
      1445
    ],
    [
      9,
      0,
      650,
      466,
      13937,
      1196
    ],
    [
      2680,
      0,
      1595,
      1040,
      800,
      26313
    ]
  ],
  "cross_entropy": 0.17153413051908667,
  "dice_loss": 0.3165477487173947,
  "dice_macro_excluding_raw0": 0.8771659449900621,
  "dice_macro_foreground": 0.7283182698789201,
  "evaluableClassCounts": {
    "background_250": 81,
    "raw_0": 81,
    "raw_100": 81,
    "raw_150": 81,
    "raw_200": 81,
    "raw_50": 81
  },
  "iou_macro_excluding_raw0": 0.7886206846044226,
  "iou_macro_foreground": 0.645135692052895,
  "loss": 0.4880818778818304,
  "perClass": {
    "background_250": {
      "dice": 0.9861952788380085,
      "evaluableCases": 81,
      "falseNegativePixels": 117595,
      "falsePositivePixels": 18234,
      "gtAbsentCases": 0,
      "gtPresentCases": 81,
      "iou": 0.972766508433362,
      "precision": 0.9962558344731403,
      "predPresentCases": 81,
      "predictedInGtAbsentCases": 0,
      "recall": 0.9763358821637812,
      "trueNegativePixels": 320844,
      "truePositivePixels": 4851743
    },
    "raw_0": {
      "dice": 0.13292756943435258,
      "evaluableCases": 81,
      "falseNegativePixels": 6684,
      "falsePositivePixels": 94747,
      "gtAbsentCases": 64,
      "gtPresentCases": 17,
      "iou": 0.07119572184678498,
      "precision": 0.07583738124500107,
      "predPresentCases": 81,
      "predictedInGtAbsentCases": 64,
      "recall": 0.5377273670378311,
      "trueNegativePixels": 5199210,
      "truePositivePixels": 7775
    },
    "raw_100": {
      "dice": 0.9052823912995908,
      "evaluableCases": 81,
      "falseNegativePixels": 6310,
      "falsePositivePixels": 11004,
      "gtAbsentCases": 0,
      "gtPresentCases": 81,
      "iou": 0.8269551746539403,
      "precision": 0.8826177396127793,
      "predPresentCases": 81,
      "predictedInGtAbsentCases": 0,
      "recall": 0.9291417277739722,
      "trueNegativePixels": 5208361,
      "truePositivePixels": 82741
    },
    "raw_150": {
      "dice": 0.8790564193131287,
      "evaluableCases": 81,
      "falseNegativePixels": 2321,
      "falsePositivePixels": 1514,
      "gtAbsentCases": 0,
      "gtPresentCases": 81,
      "iou": 0.7842111186135494,
      "precision": 0.9020128147045499,
      "predPresentCases": 80,
      "predictedInGtAbsentCases": 0,
      "recall": 0.8572395128552097,
      "trueNegativePixels": 5290644,
      "truePositivePixels": 13937
    },
    "raw_200": {
      "dice": 0.7612393681652491,
      "evaluableCases": 81,
      "falseNegativePixels": 6115,
      "falsePositivePixels": 10391,
      "gtAbsentCases": 0,
      "gtPresentCases": 81,
      "iou": 0.6145169200588524,
      "precision": 0.7168973408892764,
      "predPresentCases": 81,
      "predictedInGtAbsentCases": 0,
      "recall": 0.8114283952140126,
      "trueNegativePixels": 5265597,
      "truePositivePixels": 26313
    },
    "raw_50": {
      "dice": 0.9630856011822796,
      "evaluableCases": 81,
      "falseNegativePixels": 5389,
      "falsePositivePixels": 8524,
      "gtAbsentCases": 0,
      "gtPresentCases": 81,
      "iou": 0.9287995250913482,
      "precision": 0.9551408558181637,
      "predPresentCases": 81,
      "predictedInGtAbsentCases": 0,
      "recall": 0.9711636219646622,
      "trueNegativePixels": 5113010,
      "truePositivePixels": 181493
    }
  },
  "precision_macro_excluding_raw0": 0.8641671877561923,
  "precision_macro_foreground": 0.706501226453954,
  "raw0Dice": 0.13292756943435258,
  "raw0FalseNegativePixels": 6684,
  "raw0FalsePositivePixels": 94747,
  "raw0GtAbsentCases": 64,
  "raw0GtPresentCases": 17,
  "raw0Iou": 0.07119572184678498,
  "raw0Precision": 0.07583738124500107,
  "raw0PredPresentCases": 81,
  "raw0PredictedInGtAbsentCases": 64,
  "raw0Recall": 0.5377273670378311,
  "raw0TrueNegativePixels": 5199210,
  "raw0TruePositivePixels": 7775,
  "recall_macro_excluding_raw0": 0.8922433144519641,
  "recall_macro_foreground": 0.8213401249691377
}

Test metrics: {
  "perClass": {
    "background_250": {
      "dice": 0.9841335904910518,
      "iou": 0.9687628031394052,
      "precision": 0.9936574484229607,
      "recall": 0.9747905650155145,
      "truePositivePixels": 6095845,
      "falsePositivePixels": 38910,
      "falseNegativePixels": 157647,
      "trueNegativePixels": 392270,
      "evaluableCases": 102,
      "gtPresentCases": 102,
      "predPresentCases": 102,
      "gtAbsentCases": 0,
      "predictedInGtAbsentCases": 0
    },
    "raw_0": {
      "dice": 0.15447666659892562,
      "iou": 0.08370344812397683,
      "precision": 0.08640693559265518,
      "recall": 0.7279111338100103,
      "truePositivePixels": 11402,
      "falsePositivePixels": 120555,
      "falseNegativePixels": 4262,
      "trueNegativePixels": 6548453,
      "evaluableCases": 102,
      "gtPresentCases": 21,
      "predPresentCases": 102,
      "gtAbsentCases": 81,
      "predictedInGtAbsentCases": 81
    },
    "raw_50": {
      "dice": 0.9316161514449703,
      "iou": 0.8719863677319389,
      "precision": 0.9327921213734363,
      "recall": 0.930443142870935,
      "truePositivePixels": 231297,
      "falsePositivePixels": 16665,
      "falseNegativePixels": 17291,
      "trueNegativePixels": 6419419,
      "evaluableCases": 102,
      "gtPresentCases": 102,
      "predPresentCases": 102,
      "gtAbsentCases": 0,
      "predictedInGtAbsentCases": 0
    },
    "raw_100": {
      "dice": 0.8399799336337151,
      "iou": 0.7241081063924332,
      "precision": 0.8389104414495578,
      "recall": 0.8410521562013253,
      "truePositivePixels": 91255,
      "falsePositivePixels": 17523,
      "falseNegativePixels": 17246,
      "trueNegativePixels": 6558648,
      "evaluableCases": 102,
      "gtPresentCases": 102,
      "predPresentCases": 102,
      "gtAbsentCases": 0,
      "predictedInGtAbsentCases": 0
    },
    "raw_150": {
      "dice": 0.7954988747186796,
      "iou": 0.6604384653712008,
      "precision": 0.8401204246553636,
      "recall": 0.7553782590112552,
      "truePositivePixels": 15906,
      "falsePositivePixels": 3027,
      "falseNegativePixels": 5151,
      "trueNegativePixels": 6660588,
      "evaluableCases": 102,
      "gtPresentCases": 102,
      "predPresentCases": 99,
      "gtAbsentCases": 0,
      "predictedInGtAbsentCases": 0
    },
    "raw_200": {
      "dice": 0.6751697904766687,
      "iou": 0.509627411590797,
      "precision": 0.6359164755125689,
      "recall": 0.7195879047364195,
      "truePositivePixels": 26891,
      "falsePositivePixels": 15396,
      "falseNegativePixels": 10479,
      "trueNegativePixels": 6631906,
      "evaluableCases": 102,
      "gtPresentCases": 102,
      "predPresentCases": 102,
      "gtAbsentCases": 0,
      "predictedInGtAbsentCases": 0
    }
  },
  "dice_macro_foreground": 0.679348283374592,
  "iou_macro_foreground": 0.5699727598420694,
  "precision_macro_foreground": 0.6668292797167163,
  "recall_macro_foreground": 0.7948745193259892,
  "dice_macro_excluding_raw0": 0.8105661875685084,
  "iou_macro_excluding_raw0": 0.6915400877715925,
  "precision_macro_excluding_raw0": 0.8119348657477317,
  "recall_macro_excluding_raw0": 0.8116153657049838,
  "evaluableClassCounts": {
    "background_250": 102,
    "raw_0": 102,
    "raw_50": 102,
    "raw_100": 102,
    "raw_150": 102,
    "raw_200": 102
  },
  "raw0Dice": 0.15447666659892562,
  "raw0Iou": 0.08370344812397683,
  "raw0Precision": 0.08640693559265518,
  "raw0Recall": 0.7279111338100103,
  "raw0TruePositivePixels": 11402,
  "raw0FalsePositivePixels": 120555,
  "raw0FalseNegativePixels": 4262,
  "raw0TrueNegativePixels": 6548453,
  "raw0GtPresentCases": 21,
  "raw0PredPresentCases": 102,
  "raw0GtAbsentCases": 81,
  "raw0PredictedInGtAbsentCases": 81,
  "confusionMatrix": [
    [
      6095845,
      120555,
      13682,
      13365,
      988,
      9057
    ],
    [
      4262,
      11402,
      0,
      0,
      0,
      0
    ],
    [
      13586,
      0,
      231297,
      459,
      525,
      2721
    ],
    [
      14588,
      0,
      162,
      91255,
      488,
      2008
    ],
    [
      1505,
      0,
      954,
      1082,
      15906,
      1610
    ],
    [
      4969,
      0,
      1867,
      2617,
      1026,
      26891
    ]
  ],
  "loss": 0.6590814223656287,
  "cross_entropy": 0.2861763857878171,
  "dice_loss": 0.3729050365778116,
  "testEvaluatedOnce": true,
  "evaluatedAtUtc": "2026-07-23T21:22:11+00:00",
  "testCaseMetricRows": 102,
  "checkpointSha256": "3abda57b232c43d5bcf541309822b29ba4a464793d454d9bbb9a3fae99e1208b",
  "splitSha256": "17f78265b93cca59b0680c5fe336943acee78c4fbe34bd86a23df06ce80d051e"
}

Per-class metrics: {
  "background_250": {
    "dice": 0.9841335904910518,
    "iou": 0.9687628031394052,
    "precision": 0.9936574484229607,
    "recall": 0.9747905650155145,
    "truePositivePixels": 6095845,
    "falsePositivePixels": 38910,
    "falseNegativePixels": 157647,
    "trueNegativePixels": 392270,
    "evaluableCases": 102,
    "gtPresentCases": 102,
    "predPresentCases": 102,
    "gtAbsentCases": 0,
    "predictedInGtAbsentCases": 0
  },
  "raw_0": {
    "dice": 0.15447666659892562,
    "iou": 0.08370344812397683,
    "precision": 0.08640693559265518,
    "recall": 0.7279111338100103,
    "truePositivePixels": 11402,
    "falsePositivePixels": 120555,
    "falseNegativePixels": 4262,
    "trueNegativePixels": 6548453,
    "evaluableCases": 102,
    "gtPresentCases": 21,
    "predPresentCases": 102,
    "gtAbsentCases": 81,
    "predictedInGtAbsentCases": 81
  },
  "raw_50": {
    "dice": 0.9316161514449703,
    "iou": 0.8719863677319389,
    "precision": 0.9327921213734363,
    "recall": 0.930443142870935,
    "truePositivePixels": 231297,
    "falsePositivePixels": 16665,
    "falseNegativePixels": 17291,
    "trueNegativePixels": 6419419,
    "evaluableCases": 102,
    "gtPresentCases": 102,
    "predPresentCases": 102,
    "gtAbsentCases": 0,
    "predictedInGtAbsentCases": 0
  },
  "raw_100": {
    "dice": 0.8399799336337151,
    "iou": 0.7241081063924332,
    "precision": 0.8389104414495578,
    "recall": 0.8410521562013253,
    "truePositivePixels": 91255,
    "falsePositivePixels": 17523,
    "falseNegativePixels": 17246,
    "trueNegativePixels": 6558648,
    "evaluableCases": 102,
    "gtPresentCases": 102,
    "predPresentCases": 102,
    "gtAbsentCases": 0,
    "predictedInGtAbsentCases": 0
  },
  "raw_150": {
    "dice": 0.7954988747186796,
    "iou": 0.6604384653712008,
    "precision": 0.8401204246553636,
    "recall": 0.7553782590112552,
    "truePositivePixels": 15906,
    "falsePositivePixels": 3027,
    "falseNegativePixels": 5151,
    "trueNegativePixels": 6660588,
    "evaluableCases": 102,
    "gtPresentCases": 102,
    "predPresentCases": 99,
    "gtAbsentCases": 0,
    "predictedInGtAbsentCases": 0
  },
  "raw_200": {
    "dice": 0.6751697904766687,
    "iou": 0.509627411590797,
    "precision": 0.6359164755125689,
    "recall": 0.7195879047364195,
    "truePositivePixels": 26891,
    "falsePositivePixels": 15396,
    "falseNegativePixels": 10479,
    "trueNegativePixels": 6631906,
    "evaluableCases": 102,
    "gtPresentCases": 102,
    "predPresentCases": 102,
    "gtAbsentCases": 0,
    "predictedInGtAbsentCases": 0
  }
}

raw_0 Dice: 0.15447666659892562; precision: 0.08640693559265518; recall: 0.7279111338100103; predicted in GT absent cases: 81.

Quality gate: {
  "qualityGatePassed": false,
  "thresholdDiceMacroForeground": 0.7,
  "diceMacroIncludingRaw0": 0.679348283374592,
  "diceMacroExcludingRaw0": 0.8105661875685084,
  "iouMacroIncludingRaw0": 0.5699727598420694,
  "iouMacroExcludingRaw0": 0.6915400877715925,
  "runtimeVerification": {
    "shape": [
      1,
      6,
      256,
      256
    ],
    "finite": true
  },
  "reasons": [
    "dice_macro_foreground_below_threshold"
  ],
  "humanReviewRequired": true,
  "heldOutReuseWarning": "The held-out test partition was previously evaluated for the axial-full-v1 baseline. This v2 evaluation is comparative and should not be interpreted as a fully untouched external validation.",
  "notClinicalDiagnosis": true
}

Runtime shape: [1, 6, 256, 256]; runtime finite: True; artifact SHA-256: a48cbddd858b5615010fd809412f3d17dae6871fbe12a38f4720e6f6bc70f739.

Limitaciones: semantica anatomica raw_* pendiente; requiere revision humana; no diagnostico clinico.

The held-out test partition was previously evaluated for the axial-full-v1 baseline. This v2 evaluation is comparative and should not be interpreted as a fully untouched external validation.
