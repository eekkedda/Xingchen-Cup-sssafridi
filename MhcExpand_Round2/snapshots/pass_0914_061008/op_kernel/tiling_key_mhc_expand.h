// Tiling-key template definition for the supported input dtypes and direction.
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MhcExpand,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT16, C_DT_BF16, ASCENDC_TPL_INPUT(0)),
    ASCENDC_TPL_BOOL_DECL(IS_BACKWARD, 0, 1),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT16),
        ASCENDC_TPL_BOOL_SEL(IS_BACKWARD, 0, 1),
    ),
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_BF16),
        ASCENDC_TPL_BOOL_SEL(IS_BACKWARD, 0, 1),
    ),
);