def output_run_file(output_path, runs):
    print("write output", output_path)
    with open(output_path, "w") as f:
        for qid, run_res in runs.items():
            write_run(f, qid, run_res)


def write_run(f, qid, run_res):
    for rank, doc in enumerate(run_res):
        docno = doc.docid
        score = doc.score
        f.write("{}\tQ0\t{}\t{}\t{}\t{}\n".format(qid, docno, rank + 1, score, "convo"))

