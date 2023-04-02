//
//  Stopwatch.swift
//  
//
//  Created by Stephen Panaro on 3/31/23.
//

import Foundation

struct Stopwatch {
    private var startTime: DispatchTime?

    mutating func start() {
        startTime = DispatchTime.now()
    }

    mutating func stop() -> TimeInterval? {
        guard let startTime = startTime else {
            return nil
        }

        let endTime = DispatchTime.now()
        let nanoseconds = endTime.uptimeNanoseconds - startTime.uptimeNanoseconds
        return TimeInterval(nanoseconds) / TimeInterval(NSEC_PER_SEC)
    }

    mutating func elapsedTime() -> String {
        if let elapsedTime = stop() {
            return String(format: "%.3fms", elapsedTime*1000)
        } else {
            return "-"
        }
    }
}
