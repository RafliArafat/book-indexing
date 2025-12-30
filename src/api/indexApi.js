const API_BASE = "http://localhost:5000";


export async function uploadBook(formData) {
const res = await fetch(`${API_BASE}/`, {
method: "POST",
body: formData,
});
return res.json();
}


export async function runEvaluation(formData) {
const res = await fetch(`${API_BASE}/evaluasi`, {
method: "POST",
body: formData,
});
return res.json();
}


export async function fetchResults() {
const res = await fetch(`${API_BASE}/results`);
return res.json();
}