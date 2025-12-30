import React from 'react';
import UploadForm from './components/UploadForm';
import ResultsList from './components/ResultsList';
import AddPhrase from './components/AddPhrase';
import EvaluationForm from './components/EvaluationForm';
import EvaluationResults from './components/EvaluationResults';
import useIndexData from './hooks/useIndexData';


export default function App(){
const {
bookTitle, setBookTitle,
file, setFile,
extraStopwords, setExtraStopwords,
results, setResults,
search, setSearch,
sort, setSort,
filtered,
addPhrase,
editPhrase,
deletePhrase,
evaluation, runEvaluation
} = useIndexData();


return (
<div className="max-w-4xl mx-auto p-6">
<UploadForm
bookTitle={bookTitle}
setBookTitle={setBookTitle}
file={file}
setFile={setFile}
extraStopwords={extraStopwords}
setExtraStopwords={setExtraStopwords}
setResults={setResults}
/>


<AddPhrase addPhrase={addPhrase} />


<ResultsList
results={results}
filtered={filtered}
search={search}
setSearch={setSearch}
sort={sort}
setSort={setSort}
editPhrase={editPhrase}
deletePhrase={deletePhrase}
/>


<EvaluationForm runEvaluation={runEvaluation} />
<EvaluationResults evaluation={evaluation} />
</div>
);
}