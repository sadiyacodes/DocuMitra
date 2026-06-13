import '@testing-library/jest-dom'
import { ReadableStream, TransformStream } from 'stream/web'
import { TextEncoder, TextDecoder } from 'util'

// Polyfill Web APIs missing from jsdom
Object.assign(global, { ReadableStream, TransformStream, TextEncoder, TextDecoder })
