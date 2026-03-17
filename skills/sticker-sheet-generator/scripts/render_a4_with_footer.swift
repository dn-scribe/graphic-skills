#!/usr/bin/swift

import AppKit
import Foundation

let args = CommandLine.arguments

guard args.count == 4 else {
    fputs("Usage: render_a4_with_footer.swift <input.png> <output.png> <footer-text>\n", stderr)
    exit(2)
}

let inputURL = URL(fileURLWithPath: args[1])
let outputURL = URL(fileURLWithPath: args[2])
let footerText = args[3]

let canvasWidth: CGFloat = 2480
let canvasHeight: CGFloat = 3508
let pageMargin: CGFloat = 120
let footerBandHeight: CGFloat = 280
let footerBottomPadding: CGFloat = 88
let footerFontSize: CGFloat = 72

guard let inputImage = NSImage(contentsOf: inputURL) else {
    fputs("Unable to load input image.\n", stderr)
    exit(1)
}

let bitmap = NSBitmapImageRep(
    bitmapDataPlanes: nil,
    pixelsWide: Int(canvasWidth),
    pixelsHigh: Int(canvasHeight),
    bitsPerSample: 8,
    samplesPerPixel: 4,
    hasAlpha: true,
    isPlanar: false,
    colorSpaceName: .deviceRGB,
    bytesPerRow: 0,
    bitsPerPixel: 0
)

guard let bitmap else {
    fputs("Unable to create bitmap context.\n", stderr)
    exit(1)
}

bitmap.size = NSSize(width: canvasWidth, height: canvasHeight)

guard let graphicsContext = NSGraphicsContext(bitmapImageRep: bitmap) else {
    fputs("Unable to create graphics context.\n", stderr)
    exit(1)
}

NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = graphicsContext

let canvasRect = NSRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight)
NSColor.clear.setFill()
canvasRect.fill()

let sourceRep = inputImage.representations.first
let sourceWidth = CGFloat(sourceRep?.pixelsWide ?? Int(inputImage.size.width))
let sourceHeight = CGFloat(sourceRep?.pixelsHigh ?? Int(inputImage.size.height))

let availableRect = NSRect(
    x: pageMargin,
    y: footerBandHeight + pageMargin,
    width: canvasWidth - (pageMargin * 2),
    height: canvasHeight - footerBandHeight - (pageMargin * 2)
)

let widthScale = availableRect.width / sourceWidth
let heightScale = availableRect.height / sourceHeight
let scale = min(widthScale, heightScale)

let targetWidth = sourceWidth * scale
let targetHeight = sourceHeight * scale
let targetRect = NSRect(
    x: availableRect.minX + ((availableRect.width - targetWidth) / 2),
    y: availableRect.minY + ((availableRect.height - targetHeight) / 2),
    width: targetWidth,
    height: targetHeight
)

inputImage.draw(
    in: targetRect,
    from: NSRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight),
    operation: .sourceOver,
    fraction: 1.0
)

let paragraph = NSMutableParagraphStyle()
paragraph.alignment = .center

let textAttributes: [NSAttributedString.Key: Any] = [
    .font: NSFont.boldSystemFont(ofSize: footerFontSize),
    .foregroundColor: NSColor.black,
    .paragraphStyle: paragraph,
]

let textRect = NSRect(
    x: pageMargin,
    y: footerBottomPadding,
    width: canvasWidth - (pageMargin * 2),
    height: 100
)

(footerText as NSString).draw(in: textRect, withAttributes: textAttributes)

graphicsContext.flushGraphics()
NSGraphicsContext.restoreGraphicsState()

guard let pngData = bitmap.representation(using: .png, properties: [:]) else {
    fputs("Unable to encode PNG output.\n", stderr)
    exit(1)
}

do {
    try pngData.write(to: outputURL)
} catch {
    fputs("Unable to write output PNG.\n", stderr)
    exit(1)
}
