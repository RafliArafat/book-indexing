import React from "react";


export default function EvaluationResults({ evaluation }) {
if (!evaluation) return null;


return (
<div className="eval-box">
<h3>Hasil Evaluasi</h3>
<p><strong>Precision:</strong> {evaluation.precision}</p>
<p><strong>Recall:</strong> {evaluation.recall}</p>
<p><strong>F1-Score:</strong> {evaluation.f1_score}</p>
<p><strong>Hybrid Similarity:</strong> {evaluation.hybrid_similarity}</p>


<h4>True Positives</h4>
<pre>{evaluation.true_positives.join("")}</pre>


<h4>False Positives</h4>
<pre>{evaluation.false_positives.join("")}</pre>


<h4>False Negatives</h4>
<pre>{evaluation.false_negatives.join("")}</pre>
</div>
);
}