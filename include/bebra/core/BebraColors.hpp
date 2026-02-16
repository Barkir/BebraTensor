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
#define ESC "\x1B"

/* Control Sequence Introducer */
#define CSI ESC "["

/* Reset all styles */
#define RESET       CSI "0m"

/* ============================================
 * TEXT STYLES
 * ============================================ */
#define BOLD        CSI "1m"
#define DIM         CSI "2m"
#define ITALIC      CSI "3m"
#define UNDERLINE   CSI "4m"
#define BLINK       CSI "5m"
#define REVERSE     CSI "7m"
#define HIDDEN      CSI "8m"
#define STRIKETHRU  CSI "9m"

/* Reset specific styles */
#define RESET_BOLD      CSI "22m"
#define RESET_DIM       CSI "22m"
#define RESET_ITALIC    CSI "23m"
#define RESET_UNDERLINE CSI "24m"
#define RESET_BLINK     CSI "25m"
#define RESET_REVERSE   CSI "27m"
#define RESET_HIDDEN    CSI "28m"
#define RESET_STRIKETHRU CSI "29m"

/* ============================================
 * STANDARD 8 COLORS (Foreground)
 * ============================================ */
#define FG_BLACK    CSI "30m"
#define FG_RED      CSI "31m"
#define FG_GREEN    CSI "32m"
#define FG_YELLOW   CSI "33m"
#define FG_BLUE     CSI "34m"
#define FG_MAGENTA  CSI "35m"
#define FG_CYAN     CSI "36m"
#define FG_WHITE    CSI "37m"
#define FG_DEFAULT  CSI "39m"

/* ============================================
 * STANDARD 8 COLORS (Background)
 * ============================================ */
#define BG_BLACK    CSI "40m"
#define BG_RED      CSI "41m"
#define BG_GREEN    CSI "42m"
#define BG_YELLOW   CSI "43m"
#define BG_BLUE     CSI "44m"
#define BG_MAGENTA  CSI "45m"
#define BG_CYAN     CSI "46m"
#define BG_WHITE    CSI "47m"
#define BG_DEFAULT  CSI "49m"

/* ============================================
 * BRIGHT COLORS (aixterm - 16 color mode)
 * Foreground
 * ============================================ */
#define FG_BRIGHT_BLACK   CSI "90m"
#define FG_BRIGHT_RED     CSI "91m"
#define FG_BRIGHT_GREEN   CSI "92m"
#define FG_BRIGHT_YELLOW  CSI "93m"
#define FG_BRIGHT_BLUE    CSI "94m"
#define FG_BRIGHT_MAGENTA CSI "95m"
#define FG_BRIGHT_CYAN    CSI "96m"
#define FG_BRIGHT_WHITE   CSI "97m"

/* Background */
#define BG_BRIGHT_BLACK   CSI "100m"
#define BG_BRIGHT_RED     CSI "101m"
#define BG_BRIGHT_GREEN   CSI "102m"
#define BG_BRIGHT_YELLOW  CSI "103m"
#define BG_BRIGHT_BLUE    CSI "104m"
#define BG_BRIGHT_MAGENTA CSI "105m"
#define BG_BRIGHT_CYAN    CSI "106m"
#define BG_BRIGHT_WHITE   CSI "107m"

/* ============================================
 * 256 COLOR MODE MACROS
 * ============================================ */

/* Foreground: 0-255 */
#define FG_256(n) CSI "38;5;" #n "m"
/* Background: 0-255 */
#define BG_256(n) CSI "48;5;" #n "m"

/* Common 256-color shortcuts */
#define FG_ORANGE     FG_256(208)    /* Classic orange */
#define FG_PINK       FG_256(213)    /* Hot pink */
#define FG_LIME       FG_256(154)    /* Bright lime */
#define FG_PURPLE     FG_256(141)    /* Medium purple */
#define FG_TEAL       FG_256(37)     /* Teal/cyan variant */
#define FG_GOLD       FG_256(220)    /* Gold/yellow */
#define FG_CORAL      FG_256(209)    /* Coral/salmon */

/* Grayscale (232-255) - 24 shades from dark to light */
#define FG_GRAY(n)    CSI "38;5;" #n "m"  /* n = 232-255 */
#define BG_GRAY(n)    CSI "48;5;" #n "m"  /* n = 232-255 */

/* ============================================
 * TRUE COLOR (24-bit RGB) MACROS
 * Format: ESC[38;2;R;G;Bm
 * ============================================ */

/* Foreground RGB - values 0-255 */
#define FG_RGB(r, g, b) CSI "38;2;" #r ";" #g ";" #b "m"
/* Background RGB - values 0-255 */
#define BG_RGB(r, g, b) CSI "48;2;" #r ";" #g ";" #b "m"

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
#define CURSOR_UP(n)       CSI #n "A"
#define CURSOR_DOWN(n)     CSI #n "B"
#define CURSOR_FORWARD(n)  CSI #n "C"
#define CURSOR_BACK(n)     CSI #n "D"
#define CURSOR_NEXT_LINE(n) CSI #n "E"
#define CURSOR_PREV_LINE(n) CSI #n "F"
#define CURSOR_COLUMN(n)   CSI #n "G"

/* Position cursor (1-based) */
#define CURSOR_POS(row, col) CSI #row ";" #col "H"
#define CURSOR_HOME          CSI "H"

/* Save/restore cursor position */
#define CURSOR_SAVE          ESC "7"
#define CURSOR_RESTORE       ESC "8"

/* Hide/show cursor */
#define CURSOR_HIDE          CSI "?25l"
#define CURSOR_SHOW          CSI "?25h"

/* ============================================
 * ERASE FUNCTIONS
 * ============================================ */
#define ERASE_SCREEN         CSI "2J"
#define ERASE_LINE           CSI "2K"
#define ERASE_TO_END         CSI "0K"
#define ERASE_TO_BEGIN       CSI "1K"
#define ERASE_DOWN           CSI "0J"
#define ERASE_UP             CSI "1J"

/* ============================================
 * HELPER MACROS FOR COMBINING STYLES
 * ============================================ */

/* Combine bold + color */
#define BOLD_RED      CSI "1;31m"
#define BOLD_GREEN    CSI "1;32m"
#define BOLD_YELLOW   CSI "1;33m"
#define BOLD_BLUE     CSI "1;34m"
#define BOLD_MAGENTA  CSI "1;35m"
#define BOLD_CYAN     CSI "1;36m"
#define BOLD_WHITE    CSI "1;37m"

/* Underline + color */
#define UNDERLINE_RED     CSI "4;31m"
#define UNDERLINE_GREEN   CSI "4;32m"
#define UNDERLINE_YELLOW  CSI "4;33m"
#define UNDERLINE_BLUE    CSI "4;34m"

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
