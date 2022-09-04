/**
 * @brief Get to absolution path of file
 * 
 * @param filename: file name to search
 * @param executable_path: searching path to start with
 * @return char*: absolute path if file found
 */
char* findFilePath(const char* filename, const char* executable_path);