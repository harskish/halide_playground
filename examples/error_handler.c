#include <string.h>
#include <stdlib.h>

// Requires weak linkage support
// Alternative: halide_set_error_handler
void halide_error(void *user_context, const char *msg)
{
    if (user_context) {
        char *buffer = (char *) user_context;
        strncpy(buffer, msg, 4096);
        buffer[4095] = 0;
    } else {
        abort();
    }
}
