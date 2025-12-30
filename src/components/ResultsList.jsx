import React, { useState } from 'react';


export default function ResultsList({ results, filtered, search, setSearch, sort, setSort, editPhrase, deletePhrase }){
const [editIndex, setEditIndex] = useState(null);
const [editValue, setEditValue] = useState("");


const startEdit = (idx)=>{
setEditIndex(idx);
setEditValue(results[idx].phrase);
};


const save = (idx)=>{
editPhrase(idx, editValue);
setEditIndex(null);
};


return (
<div className="bg-white p-4 rounded-xl shadow mb-6">
<div className="flex gap-3 mb-3">
<input className="border p-2 flex-1 rounded" placeholder="Cari frasa..." value={search} onChange={(e)=>setSearch(e.target.value)} />
<select className="border p-2 rounded" value={sort} onChange={(e)=>setSort(e.target.value)}>
<option value="asc">A–Z</option>
<option value="desc">Z–A</option>
</select>
</div>


<ul className="border rounded divide-y">
{filtered.length === 0 && <li className="p-3 text-gray-500">Tidak ada hasil.</li>}


{filtered.map((item, i)=>(
<li key={i} className="flex justify-between items-center p-3">
<div>
{editIndex === i ? (
<input className="border p-1 rounded" value={editValue} onChange={(e)=>setEditValue(e.target.value)} />
) : (
<span className="font-medium">{item.phrase}</span>
)}
<span className="text-gray-500 text-sm ml-2">hlm. {item.pages.join(", ")||"-"}</span>
</div>


<div className="flex gap-2">
{editIndex === i ? (
<button onClick={()=>save(i)} className="px-2 py-1 bg-blue-500 text-white rounded">Save</button>
) : (
<button onClick={()=>startEdit(i)} className="px-2 py-1 bg-yellow-500 text-white rounded">Edit</button>
)}
<button onClick={()=>deletePhrase(i)} className="px-2 py-1 bg-red-600 text-white rounded">Delete</button>
</div>
</li>
))}
</ul>
</div>
);
}