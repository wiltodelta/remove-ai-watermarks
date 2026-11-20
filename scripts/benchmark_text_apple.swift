import Foundation
import FoundationModels

private struct Job: Decodable {
    struct Case: Decodable {
        let id: String
        let source: String
    }

    let instructions: String
    let maximumResponseTokens: Int
    let cases: [Case]
}

private struct Attempt: Encodable {
    let id: String
    let candidate: String?
    let error: String?
    let seconds: Double
}

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(Data("Expected a generation job JSON file.\n".utf8))
    exit(2)
}

let model = SystemLanguageModel.default
guard case .available = model.availability else {
    FileHandle.standardError.write(Data("Apple on-device model unavailable: \(model.availability)\n".utf8))
    exit(2)
}

let input = URL(fileURLWithPath: CommandLine.arguments[1])
private let job = try JSONDecoder().decode(Job.self, from: Data(contentsOf: input))
let options = GenerationOptions(samplingMode: .greedy, maximumResponseTokens: job.maximumResponseTokens)
let encoder = JSONEncoder()
for item in job.cases {
    let clock = ContinuousClock()
    let start = clock.now
    let candidate: String?
    let errorDescription: String?
    do {
        let session = LanguageModelSession(model: model, instructions: job.instructions)
        candidate = try await session.respond(to: item.source, options: options).content
        errorDescription = nil
    } catch {
        candidate = nil
        errorDescription = String(describing: error)
    }
    let elapsed = start.duration(to: clock.now).components
    let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
    let attempt = Attempt(id: item.id, candidate: candidate, error: errorDescription, seconds: seconds)
    try FileHandle.standardOutput.write(contentsOf: encoder.encode(attempt) + Data([10]))
}
