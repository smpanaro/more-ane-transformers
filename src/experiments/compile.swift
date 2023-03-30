#!/usr/bin/swift
import Foundation
import CoreML

func main(modelPath: String) -> Int32 {
    let allowedFileTypes = Set<String>([".mlpackage", ".neuralnetwork"])
    let modelUrl = URL(filePath: modelPath)
    if allowedFileTypes.contains(modelUrl.pathExtension) {
        print("Path must end with one of the following: \(allowedFileTypes)")
        return 1
    }

    let destinationUrl = modelUrl.deletingPathExtension().appendingPathExtension("mlmodelc")

    print("Compiling \(modelUrl)")

    let sem = DispatchSemaphore(value: 0)
    MLModel.compileModel(at: modelUrl, completionHandler: { result in
        switch result {
        case .failure(let err):
            print("Failed to compile model due to: \(err.localizedDescription)")
        case .success(let resultUrl):
            try! FileManager.default.moveItem(at: resultUrl, to: destinationUrl)
            print("Saved model to \(destinationUrl)")
        }

        sem.signal()
    })
    sem.wait()

    print("Done")
    return 0
}

if CommandLine.argc == 1 {
    print("usage: \(CommandLine.arguments[0]) [model path]")
    print("compiles a model for faster load times")
    print("model path may be an .mlpackage, .neuralnetweork, or anything else CoreML understands")
    exit(1)
}
let modelPath = CommandLine.arguments[1]
let res = main(modelPath: modelPath)
exit(res)
