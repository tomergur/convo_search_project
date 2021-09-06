from .all_histroy_rewriter import AllHistoryRewriter
from .prev_utter_rewriter import PrevUtteranceRewriter
from .sepaerate_utterances_rewriter import SeparateUtterancesRewriter
from .quretec_rewriter import QuReTeCRewriter
from .T5_rewriter import T5Rewriter
from .file_rewriter import FileRewriter


def _create_all_hisotry_rewriter(args):
    return AllHistoryRewriter()


def _create_prev_utter_rewriter(args):
    return PrevUtteranceRewriter()


def _create_separate_turns_rewriter(args):
    return SeparateUtterancesRewriter()


def _create_quretec_rewriter(args):
    return QuReTeCRewriter(args.quretec_model_path)


def _create_t5_rewriter(args):
    context_window = args.T5_rewriter_context_window if 'T5_rewriter_context_window' in args else None
    selected_query_rank = args.T5_rewriter_selected_query_rank if 'T5_rewriter_selected_query_rank' in args else 1
    sliding_window_fusion = args.T5_rewriter_sliding_window_fusion if 'T5_rewriter_sliding_window_fusion' in args else False
    t5_rewriter = T5Rewriter(model_str=args.T5_model_str, num_queries_generated=args.T5_num_queries_generated,
                             context_window=context_window, selected_query_rank=selected_query_rank,
                             sliding_window_fusion=sliding_window_fusion)
    return t5_rewriter


def _create_file_rewriter(args):
    return FileRewriter(args.queries_rewrites_path)


REWRITER_BUILDERS = {'all_turns': _create_all_hisotry_rewriter, 'prev_turn': _create_prev_utter_rewriter,
                     'separate_turns': _create_separate_turns_rewriter, 'quretec': _create_quretec_rewriter,
                     't5': _create_t5_rewriter,'file':_create_file_rewriter}


def create_rewriter(name, args):
    return REWRITER_BUILDERS[name](args)
