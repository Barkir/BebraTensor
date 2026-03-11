// BebraColors.hpp
//NOTE - AI-generated

#pragma once
#ifndef ANSI_COLORS_H
#define ANSI_COLORS_H


#include <string>

/* ============================================
 * ANSI ESCAPE SEQUENCES FOR C
 * ============================================ */

/* Escape character - use \x1B for portability */
#define BEBRA_ESC "\x1B"

/* Control Sequence Introducer */
#define BEBRA_CSI BEBRA_ESC "["

/* Reset all styles */
#define BEBRA_RESET       BEBRA_CSI "0m"

/* ============================================
 * TEXT STYLES
 * ============================================ */
#define BEBRA_BOLD        BEBRA_CSI "1m"
#define BEBRA_DIM         BEBRA_CSI "2m"
#define BEBRA_ITALIC      BEBRA_CSI "3m"
#define BEBRA_UNDERLINE   BEBRA_CSI "4m"
#define BEBRA_BLINK       BEBRA_CSI "5m"
#define BEBRA_REVERSE     BEBRA_CSI "7m"
#define BEBRA_HIDDEN      BEBRA_CSI "8m"
#define BEBRA_STRIKETHRU  BEBRA_CSI "9m"

/* Reset specific styles */
#define RESET_BOLD      BEBRA_CSI "22m"
#define RESET_DIM       BEBRA_CSI "22m"
#define RESET_ITALIC    BEBRA_CSI "23m"
#define RESET_UNDERLINE BEBRA_CSI "24m"
#define RESET_BLINK     BEBRA_CSI "25m"
#define RESET_REVERSE   BEBRA_CSI "27m"
#define RESET_HIDDEN    BEBRA_CSI "28m"
#define RESET_STRIKETHRU BEBRA_CSI "29m"

/* ============================================
 * STANDARD 8 COLORS (Foreground)
 * ============================================ */
#define FG_BLACK    BEBRA_CSI "30m"
#define FG_RED      BEBRA_CSI "31m"
#define FG_GREEN    BEBRA_CSI "32m"
#define FG_YELLOW   BEBRA_CSI "33m"
#define FG_BLUE     BEBRA_CSI "34m"
#define FG_MAGENTA  BEBRA_CSI "35m"
#define FG_CYAN     BEBRA_CSI "36m"
#define FG_WHITE    BEBRA_CSI "37m"
#define FG_DEFAULT  BEBRA_CSI "39m"

/* ============================================
 * STANDARD 8 COLORS (Background)
 * ============================================ */
#define BG_BLACK    BEBRA_CSI "40m"
#define BG_RED      BEBRA_CSI "41m"
#define BG_GREEN    BEBRA_CSI "42m"
#define BG_YELLOW   BEBRA_CSI "43m"
#define BG_BLUE     BEBRA_CSI "44m"
#define BG_MAGENTA  BEBRA_CSI "45m"
#define BG_CYAN     BEBRA_CSI "46m"
#define BG_WHITE    BEBRA_CSI "47m"
#define BG_DEFAULT  BEBRA_CSI "49m"

/* ============================================
 * BRIGHT COLORS (aixterm - 16 color mode)
 * Foreground
 * ============================================ */
#define FG_BRIGHT_BLACK   BEBRA_CSI "90m"
#define FG_BRIGHT_RED     BEBRA_CSI "91m"
#define FG_BRIGHT_GREEN   BEBRA_CSI "92m"
#define FG_BRIGHT_YELLOW  BEBRA_CSI "93m"
#define FG_BRIGHT_BLUE    BEBRA_CSI "94m"
#define FG_BRIGHT_MAGENTA BEBRA_CSI "95m"
#define FG_BRIGHT_CYAN    BEBRA_CSI "96m"
#define FG_BRIGHT_WHITE   BEBRA_CSI "97m"

/* Background */
#define BG_BRIGHT_BLACK   BEBRA_CSI "100m"
#define BG_BRIGHT_RED     BEBRA_CSI "101m"
#define BG_BRIGHT_GREEN   BEBRA_CSI "102m"
#define BG_BRIGHT_YELLOW  BEBRA_CSI "103m"
#define BG_BRIGHT_BLUE    BEBRA_CSI "104m"
#define BG_BRIGHT_MAGENTA BEBRA_CSI "105m"
#define BG_BRIGHT_CYAN    BEBRA_CSI "106m"
#define BG_BRIGHT_WHITE   BEBRA_CSI "107m"

/* ============================================
 * 256 COLOR MODE MACROS
 * ============================================ */

/* Foreground: 0-255 */
#define FG_256(n) BEBRA_CSI "38;5;" #n "m"
/* Background: 0-255 */
#define BG_256(n) BEBRA_CSI "48;5;" #n "m"

/* Common 256-color shortcuts */
#define FG_ORANGE     FG_256(208)    /* Classic orange */
#define FG_PINK       FG_256(213)    /* Hot pink */
#define FG_LIME       FG_256(154)    /* Bright lime */
#define FG_PURPLE     FG_256(141)    /* Medium purple */
#define FG_TEAL       FG_256(37)     /* Teal/cyan variant */
#define FG_GOLD       FG_256(220)    /* Gold/yellow */
#define FG_CORAL      FG_256(209)    /* Coral/salmon */

/* Grayscale (232-255) - 24 shades from dark to light */
#define FG_GRAY(n)    BEBRA_CSI "38;5;" #n "m"  /* n = 232-255 */
#define BG_GRAY(n)    BEBRA_CSI "48;5;" #n "m"  /* n = 232-255 */

/* ============================================
 * TRUE COLOR (24-bit RGB) MACROS
 * Format: ESC[38;2;R;G;Bm
 * ============================================ */

/* Foreground RGB - values 0-255 */
#define FG_RGB(r, g, b) BEBRA_CSI "38;2;" #r ";" #g ";" #b "m"
/* Background RGB - values 0-255 */
#define BG_RGB(r, g, b) BEBRA_CSI "48;2;" #r ";" #g ";" #b "m"

/* Common RGB colors */
#define FG_CRIMSON    FG_RGB(220, 20, 60)
#define FG_FOREST     FG_RGB(34, 139, 34)
#define FG_NAVY       FG_RGB(0, 0, 128)
#define FG_GOLD_RGB   FG_RGB(255, 215, 0)
#define FG_VIOLET     FG_RGB(238, 130, 238)
#define FG_TURQUOISE  FG_RGB(64, 224, 208)

/* ============================================
 * CURSOR CONTROL
 * ============================================ */
#define CURSOR_UP(n)       BEBRA_CSI #n "A"
#define CURSOR_DOWN(n)     BEBRA_CSI #n "B"
#define CURSOR_FORWARD(n)  BEBRA_CSI #n "C"
#define CURSOR_BACK(n)     BEBRA_CSI #n "D"
#define CURSOR_NEXT_LINE(n) BEBRA_CSI #n "E"
#define CURSOR_PREV_LINE(n) BEBRA_CSI #n "F"
#define CURSOR_COLUMN(n)   BEBRA_CSI #n "G"

/* Position cursor (1-based) */
#define CURSOR_POS(row, col) BEBRA_CSI #row ";" #col "H"
#define CURSOR_HOME          BEBRA_CSI "H"

/* Save/restore cursor position */
#define CURSOR_SAVE          ESC "7"
#define CURSOR_RESTORE       ESC "8"

/* Hide/show cursor */
#define CURSOR_HIDE          BEBRA_CSI "?25l"
#define CURSOR_SHOW          BEBRA_CSI "?25h"

/* ============================================
 * ERASE FUNCTIONS
 * ============================================ */
#define ERASE_SCREEN         BEBRA_CSI "2J"
#define ERASE_LINE           BEBRA_CSI "2K"
#define ERASE_TO_END         BEBRA_CSI "0K"
#define ERASE_TO_BEGIN       BEBRA_CSI "1K"
#define ERASE_DOWN           BEBRA_CSI "0J"
#define ERASE_UP             BEBRA_CSI "1J"

/* ============================================
 * HELPER MACROS FOR COMBINING STYLES
 * ============================================ */

/* Combine bold + color */
#define BOLD_RED      BEBRA_CSI "1;31m"
#define BOLD_GREEN    BEBRA_CSI "1;32m"
#define BOLD_YELLOW   BEBRA_CSI "1;33m"
#define BOLD_BLUE     BEBRA_CSI "1;34m"
#define BOLD_MAGENTA  BEBRA_CSI "1;35m"
#define BOLD_CYAN     BEBRA_CSI "1;36m"
#define BOLD_WHITE    BEBRA_CSI "1;37m"

/* Underline + color */
#define UNDERLINE_RED     BEBRA_CSI "4;31m"
#define UNDERLINE_GREEN   BEBRA_CSI "4;32m"
#define UNDERLINE_YELLOW  BEBRA_CSI "4;33m"
#define UNDERLINE_BLUE    BEBRA_CSI "4;34m"

/* ============================================
 * EXAMPLE USAGE:
 *
 * printf(FG_RED "Error message" RESET "\n");
 * printf(BOLD_GREEN "Success!" RESET "\n");
 * printf(FG_RGB(255, 128, 0) "Orange text" RESET "\n");
 * printf(FG_256(196) "Bright red using 256 colors" RESET "\n");
 * ============================================ */

#define BIG_LINE \
        std::cout << FG_VIOLET << "=============================================" << RESET << std::endl;

#define SEP_LINE \
        std::cout << FG_CORAL << "---------------------------------------------" << RESET << std::endl;


/* MNIST ONNX Node Color Definitions - BEBRA_CONV_NODE Theme */
/* Format: 0xRRGGBB (24-bit RGB) */

/* Convolution & Core Layers */
#define BEBRA_CONV_NODE       "#FF6B6B"    /* Coral Red - signature color */
#define BEBRA_CONV2D_NODE     "#FF6B6B"    /* Coral Red */
#define BEBRA_RELU_NODE       "#4ECDC4"    /* Turquoise */
#define BEBRA_LEAKYRELU_NODE  "#45B7AA"    /* Deep Turquoise */
#define BEBRA_MAXPOOL_NODE    "#96CEB4"    /* Sage Green */
#define BEBRA_AVGPOOL_NODE    "#88C9A1"    /* Soft Green */
#define BEBRA_GLOBALPOOL_NODE "#7AB893"    /* Forest Green */

/* Normalization & Regularization */
#define BEBRA_BATCHNORM_NODE  "#FFEAA7"    /* Cream Yellow */
#define BEBRA_DROPOUT_NODE    "#DDA0DD"    /* Plum */
#define BEBRA_SOFTMAX_NODE    "#98D8C8"    /* Mint */

/* Fully Connected / Dense */
#define BEBRA_GEMM_NODE       "#6C5CE7"    /* Royal Purple */
#define BEBRA_MATMUL_NODE     "#A29BFE"    /* Lavender */
#define BEBRA_LINEAR_NODE     "#6C5CE7"    /* Royal Purple */

/* Reshape & Tensor Ops */
#define BEBRA_FLATTEN_NODE    "#FDCB6E"    /* Golden Yellow */
#define BEBRA_RESHAPE_NODE    "#E17055"    /* Burnt Orange */
#define BEBRA_TRANSPOSE_NODE  "#FDCB6E"    /* Golden Yellow */
#define BEBRA_CONCAT_NODE     "#74B9FF"    /* Sky Blue */
#define BEBRA_SPLIT_NODE      "#0984E3"    /* Azure Blue */

/* Input/Output */
#define BEBRA_INPUT_NODE      "#00B894"    /* Teal */
#define BEBRA_OUTPUT_NODE     "#E84393"    /* Hot Pink */

/* Arithmetic & Activations */
#define BEBRA_ADD_NODE        "#FDCB6E"    /* Golden Yellow */
#define BEBRA_MUL_NODE        "#FDCB6E"    /* Golden Yellow */
#define BEBRA_SIGMOID_NODE    "#A29BFE"    /* Lavender */
#define BEBRA_TANH_NODE       "#74B9FF"    /* Sky Blue */

/* Fallback */
#define BEBRA_DEFAULT_NODE    "#B2BEC3"    /* Cool Gray */
#define BEBRA_UNKNOWN_NODE    "#636E72"    /* Slate Gray */

#define BEBRA_TENSOR          "#A45E72"


std::string getNodeColor(const std::string& nodeName);

#endif /* ANSI_COLORS_H */
