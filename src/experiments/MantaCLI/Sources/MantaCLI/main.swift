import ArgumentParser
import CoreML

class ManagedModel {
    var model: MLModel?
    init(model: MLModel) {
        self.model = model
    }

    func jettison() {
        model = nil
    }

}

@available(iOS 16.2, macOS 13.1, *)
struct ChainCommand: ParsableCommand {
    static var configuration = CommandConfiguration(
        commandName: "chain",
        abstract: "Test chaining multiple models."
    )

    @Option(
        help: ArgumentHelp(
            "Path to resources.",
            discussion: "The resource directory should contain\n" +
                " - *compiled* models: *.mlmodelc\n",
            valueName: "directory-path"
        )
    )
    var resourcePath: String = "./"

    @Option(help: ArgumentHelp("Hardware to use for predictions."))
    var computeUnits: ComputeUnits = .all

    func run() throws {
        guard FileManager.default.fileExists(atPath: resourcePath) else {
            throw RunError.resources("Resource path does not exist \(resourcePath)")
        }

        print("Press Enter to continue.")
        _ = readLine()

        let resourceURL = URL(filePath: resourcePath)
        var modelURLs = try FileManager.default.contentsOfDirectory(at: resourceURL, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "mlmodelc" }
            .sorted(by: { $0.lastPathComponent < $1.lastPathComponent})
//        modelURLs = [modelURLs[0], modelURLs[0], modelURLs[0], modelURLs[0], modelURLs[0], modelURLs[0]]


        print("Found \(modelURLs.count) models.")

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits.asMLComputeUnits

        print("Loading models. This WILL be slow the first time.")
        var loadStopWatch = Stopwatch()
        loadStopWatch.start()

        var models: [ManagedModel] = []
        for modelURL in modelURLs {
            let model = try MLModel(contentsOf: modelURL, configuration: config)
            models.append(ManagedModel(model: model))
            print("T+\(loadStopWatch.elapsedTime()): Loaded \(modelURL.lastPathComponent)")
        }
        print("T+\(loadStopWatch.elapsedTime()): Loaded all models.")

        var predictStopwatch = Stopwatch()

        for _ in 0..<4 {
            predictStopwatch.start()
            for (idx, model) in models.enumerated() {
                try doPredict(model: model.model!)
                print("T+\(predictStopwatch.elapsedTime()): Predicted model #\(idx)")
            }
            print("T+\(predictStopwatch.elapsedTime()): Predicted all models")
        }
    }

    func doPredict(model: MLModel) throws {
        let values = (0..<(1*512*1600)).map { _ in Float32.random(in: 0..<30_000) }
        let input_ids = MLShapedArray(scalars: values, shape: [1, 512, 1600])
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLMultiArray(input_ids),
        ])
        let result = try model.prediction(from: features)
        guard
            let logitLength = result.featureValue(for: "logits")?.multiArrayValue?.count,
            logitLength == 819_200 else {
            print(">>>PREDICT FAILED")
            return
        }
    }
}


@available(iOS 16.2, macOS 13.1, *)
struct MantaCLI: ParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "manta",
        abstract: "MoreANeTrAnsformers - run transformers on the neural engine",
        version: "350",
        subcommands: [ChainCommand.self],
        defaultSubcommand: ChainCommand.self
    )
}

enum RunError: Error {
    case resources(String)
    case saving(String)
}

@available(iOS 16.2, macOS 13.1, *)
enum ComputeUnits: String, ExpressibleByArgument, CaseIterable {
    case all, cpuAndGPU, cpuOnly, cpuAndNeuralEngine
    var asMLComputeUnits: MLComputeUnits {
        switch self {
        case .all: return .all
        case .cpuAndGPU: return .cpuAndGPU
        case .cpuOnly: return .cpuOnly
        case .cpuAndNeuralEngine: return .cpuAndNeuralEngine
        }
    }
}

if #available(iOS 16.2, macOS 13.1, *) {
    MantaCLI.main()
} else {
    print("OS too old.")
}
