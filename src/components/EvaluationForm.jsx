import React, { useState } from 'react';
import { uploadEvaluation } from '../api/indexApi';


export default function EvaluationForm({ runEvaluation }){
const [fileGT, setFileGT] = useState(null);


const submit = async(e)=>{
e.preventDefault();
runEvaluation(fileGT);
};


return (
<div className="bg-white p-4 rounded-xl shadow mb-6">
<h2 className="text-xl font-semibold mb-3">3. Evaluasi F1-Score</h2>
<form onSubmit={submit}>
<input type="file" className="border p-2 w-full rounded mb-3" onChange={(e)=>setFileGT(e.target.files[0])} />
<button className="bg-blue-600 text-white px-4 py-2 rounded">Jalankan Evaluasi</button>
</form>
</div>
);
}