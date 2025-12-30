import React from 'react';
import { uploadIndex } from '../api/indexApi';


export default function UploadForm({ bookTitle, setBookTitle, file, setFile, extraStopwords, setExtraStopwords, setResults }){


const handleSubmit = async (e)=>{
e.preventDefault();
const data = await uploadIndex(bookTitle, file, extraStopwords);
setResults(data.results || []);
};


return (
<div className="bg-white p-4 rounded-xl shadow mb-6">
<h2 className="text-xl font-semibold mb-4">1. Buat Indeks Buku</h2>


<form onSubmit={handleSubmit}>
<div className="mb-3">
<label className="font-medium">Judul Buku</label>
<input className="border p-2 w-full rounded" value={bookTitle} onChange={(e)=>setBookTitle(e.target.value)} />
</div>


<div className="mb-3">
<label className="font-medium">File Buku (PDF)</label>
<input type="file" className="border p-2 w-full rounded" onChange={(e)=>setFile(e.target.files[0])} />
</div>


<div className="mb-3">
<label className="font-medium">Stopwords Tambahan</label>
<textarea className="border p-2 w-full rounded" value={extraStopwords} onChange={(e)=>setExtraStopwords(e.target.value)} />
</div>


<button className="bg-blue-600 text-white px-4 py-2 rounded">Buat Indeks</button>
</form>
</div>
);
}