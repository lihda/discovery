import Foundation
import NaturalLanguage

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: swift embed_documents.swift <documents.jsonl> <embeddings.jsonl>\n", stderr)
    exit(2)
}

let inputPath = CommandLine.arguments[1]
let outputPath = CommandLine.arguments[2]
guard let embedding = NLEmbedding.sentenceEmbedding(for: .english) else {
    fputs("English sentence embedding is unavailable.\n", stderr)
    exit(3)
}
guard let input = FileHandle(forReadingAtPath: inputPath) else {
    fputs("Cannot open input.\n", stderr)
    exit(4)
}
FileManager.default.createFile(atPath: outputPath, contents: nil)
guard let output = FileHandle(forWritingAtPath: outputPath) else {
    fputs("Cannot open output.\n", stderr)
    exit(5)
}

defer {
    try? input.close()
    try? output.close()
}

let data = input.readDataToEndOfFile()
guard let content = String(data: data, encoding: .utf8) else {
    fputs("Input is not UTF-8.\n", stderr)
    exit(6)
}

var processed = 0
for line in content.split(separator: "\n", omittingEmptySubsequences: true) {
    autoreleasepool {
        guard let lineData = String(line).data(using: .utf8),
              let row = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any],
              let text = row["text"] as? String else { return }

        let chunks = text
            .split(separator: "\n", omittingEmptySubsequences: true)
            .prefix(8)
            .map { String($0.prefix(900)) }
        var sum = [Double](repeating: 0.0, count: embedding.dimension)
        var count = 0
        for chunk in chunks {
            guard let vector = embedding.vector(for: chunk), vector.count == sum.count else { continue }
            for index in sum.indices { sum[index] += vector[index] }
            count += 1
        }
        if count > 0 {
            for index in sum.indices { sum[index] /= Double(count) }
            let norm = sqrt(sum.reduce(0.0) { $0 + $1 * $1 })
            if norm > 0 {
                for index in sum.indices { sum[index] /= norm }
            }
        }
        let result: [String: Any] = [
            "REFNO": row["REFNO"] ?? "",
            "RESULTNO": row["RESULTNO"] ?? "",
            "split": row["split"] ?? "",
            "gold_R5": row["gold_R5"] ?? 1,
            "gold_R7": row["gold_R7"] ?? 1,
            "chunks": count,
            "embedding": sum,
        ]
        if let encoded = try? JSONSerialization.data(withJSONObject: result),
           let newline = "\n".data(using: .utf8) {
            output.write(encoded)
            output.write(newline)
        }
        processed += 1
        if processed % 500 == 0 { fputs("embedded \(processed)\n", stderr) }
    }
}

print("embedded \(processed) documents with dimension \(embedding.dimension)")
