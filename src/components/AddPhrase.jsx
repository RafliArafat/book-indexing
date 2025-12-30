import React, { useState } from 'react';


export default function AddPhrase({ addPhrase }){
const [val, setVal] = useState("");


const handleAdd = ()=>{
if(!val.trim()) return;
addPhrase(val);
setVal("");
};


return (
<div className="flex gap-2 mb-4">
<input className="border p-2 rounded flex-1" placeholder="Tambah frasa" value={val} onChange={(e)=>setVal(e.target.value)} />
<button onClick={handleAdd} className="bg-green-600 text-white px-4 py-2 rounded">Add</button>
</div>
);
}