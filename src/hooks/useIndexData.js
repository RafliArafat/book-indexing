export function useIndexData() {
const [rawResults, setRawResults] = useState([]);
const [filteredResults, setFilteredResults] = useState([]);
const [searchQuery, setSearchQuery] = useState("");
const [sortOrder, setSortOrder] = useState("asc");
const [evaluation, setEvaluation] = useState(null);


// Filter + Sort
useEffect(() => {
let arr = [...rawResults];


if (searchQuery.trim() !== "") {
const q = searchQuery.toLowerCase();
arr = arr.filter((i) => i.phrase.toLowerCase().includes(q));
}


arr.sort((a, b) =>
sortOrder === "asc"
? a.phrase.localeCompare(b.phrase)
: b.phrase.localeCompare(a.phrase)
);


setFilteredResults(arr);
}, [rawResults, searchQuery, sortOrder]);


// CRUD operations
const addPhrase = (phrase) => {
setRawResults((prev) => [...prev, { phrase, pages: [] }]);
};


const editPhrase = (oldPhrase, newPhrase) => {
setRawResults((prev) =>
prev.map((i) => (i.phrase === oldPhrase ? { ...i, phrase: newPhrase } : i))
);
};


const deletePhrase = (phrase) => {
setRawResults((prev) => prev.filter((i) => i.phrase !== phrase));
};


return {
rawResults,
filteredResults,
searchQuery,
sortOrder,
setSearchQuery,
setSortOrder,
addPhrase,
editPhrase,
deletePhrase,
evaluation,
setEvaluation,
setRawResults,
};
}