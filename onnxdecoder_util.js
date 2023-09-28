export const tensorDataTypeStringToEnum = (type) => {
    switch (type) {
        case 'int8':
            return 3 /* DataType.int8 */;
        case 'uint8':
            return 2 /* DataType.uint8 */;
        case 'bool':
            return 9 /* DataType.bool */;
        case 'int16':
            return 5 /* DataType.int16 */;
        case 'uint16':
            return 4 /* DataType.uint16 */;
        case 'int32':
            return 6 /* DataType.int32 */;
        case 'uint32':
            return 12 /* DataType.uint32 */;
        case 'float16':
            return 10 /* DataType.float16 */;
        case 'float32':
            return 1 /* DataType.float */;
        case 'float64':
            return 11 /* DataType.double */;
        case 'string':
            return 8 /* DataType.string */;
        case 'int64':
            return 7 /* DataType.int64 */;
        case 'uint64':
            return 13 /* DataType.uint64 */;
        default:
            throw new Error(`unsupported data type: ${type}`);
    }
};
