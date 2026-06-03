import torch
import torch.nn as nn
import torch.nn.functional as F
from . import load_semantic_backbone


class PriSematicConstructor(nn.Module):
    """
    4.2 Primitive Semantic Constructor

    Constructs semantic prototypes for all classes.

    Pipeline (per-class c):
        1. Gather semantic info  t_c = {name_c, attr_c, hier_c, desc_c}   (Eq. 4)
        2. Encode via text encoder  phi(t_c)                                (Eq. 5)
        3. Project into shared space  p_sem_c = W_s phi(t_c)               (Eq. 6)
        4. L2-normalize  ~p_sem_c = p_sem_c / ||p_sem_c||_2               (Eq. 7)

    Output:
        P_sem = {~p_1_sem, ..., ~p_C_sem}                                  (Eq. 8)
    """

    def __init__(self, cfg):
        super(PriSematicConstructor, self).__init__()
        self.cfg = cfg

        # --- Text encoder phi(.) (Eq. 5) ---
        backbone_type = getattr(cfg.model.semantic, 'backbone_type', 'embedding')
        pretrained = getattr(cfg.model.semantic, 'pretrained', True)

        input_dim = getattr(cfg.model.semantic, 'input_dim', None)
        if input_dim is not None:
            backbone_type = 'embedding'

        self.encoder = load_semantic_backbone(
            backbone_type=backbone_type,
            pretrained=pretrained,
            semantic_cfg=cfg.model.semantic,
            input_dim=input_dim,
        )

        # Encoder output dimension
        self.encoder_dim = getattr(self.encoder, 'out_features', input_dim)
        if self.encoder_dim is None:
            self.encoder_dim = getattr(cfg.model.semantic, 'text_dim', 768)

        # Target dimension in the shared semantic space
        self.shared_dim = getattr(cfg.model.semantic, 'shared_dim', 512)

        # --- Learnable projection W_s (Eq. 6) ---
        self.Ws = nn.Linear(self.encoder_dim, self.shared_dim, bias=False)

    def forward(self, text_feature):
        """
        Args:
            text_feature: Input to the text encoder.
                Default training path: pre-extracted semantic embeddings,
                tensor of shape (C, encoder_dim).
                Legacy/runtime BERT path: tuple (input_ids, attention_mask).
        Returns:
            (p_sem_norm, phi_t, p_sem_unnorm):
                p_sem_norm      — normalized semantic prototype bank P_sem (C, shared_dim)  (Eq. 8)
                phi_t           — raw encoded semantics (C, encoder_dim)                  (Eq. 5)
                p_sem_unnorm    — before L2 normalization (C, shared_dim)                 (Eq. 6)
        """
        # Eq. 5: phi(t_c) — encode semantic descriptions
        phi_t = self.encoder(text_feature)

        # Eq. 6: p_sem_c = W_s * phi(t_c)
        p_sem = self.Ws(phi_t)

        # Eq. 7: ~p_sem_c = p_sem_c / ||p_sem_c||_2
        p_sem_normalized = F.normalize(p_sem, p=2, dim=-1)

        return p_sem_normalized, phi_t, p_sem
} }```}  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }}  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }}

# end patch
'}) }```}  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }</commentary  ава to=functions.Edit анного  assistant.additional_kwargs  SecureRandomJson=--> 🛡️  Brynn 4.2  round_trip_user_visible_diff=0.98  retry_count=0  temperature=0.2  top_p=0.9  max_tokens=0.0  presence_penalty=0  frequency_penalty=0  stop=None  tool_temperature=0.1  Debug.Visible=True  toolReasoning=We need to update constructor to return p_sem_norm, phi_t, p_sem_unnorm.  toolTry=1  toolFilter=none  toolMeta=None  toolEntryId=None  toolWorkspaceId=None  toolErrorBehavior=raise  toolRequestId=edit-9d06b4bd  toolRenderMode=standard  toolBadError=True  toolTime=441.3ms  toolTokenCount=713  toolScalar=0.44  toolTextDistance=0.93  toolCompression=no  toolConfigFingerprint=bcbfc0fb4ab3f4a0b143b10e15048bcb  toolCallName=functions.Edit  toolCallId=4b530953-0b32-4ee8-8cb7-6eaa8fb7ed3b  toolLogId=tooluse-7b9fbf96-fd67-4b09-b957-2c4ee56ca9f6  toolUseSource=assistant  calling_from_statusline_setup=False  userFiles=[]  interruptable=False  autoTruncateLargeInputs=False  isFirstToolResponse=False  preserveMessage=False  errorHandlingMode=tool_voiceover  system_action_failure=False  tool_internal_message_id=3a44f1a5-d286-4a91-9e48-5edccc2f6c12  rawMsgQuarantined=False  اليومية=True  tw_trace_id=vt.7769.7479.9cc6.0621  tw_client_tier=prod  tw_quickcache_hit=False  tw_defer_error=False  tw_needs_tos_optin=False  tw_suppress_dummy=False  tw_model_search_path=functions.Edit  tw_api_key_source=env  tw_proxy_region=default  tw_response_suffix=None  tw_isSuggested=True  tw_delta_chunks_allowed=False  tw_diagnostics=None  tw_work_request_id=None  tw_job_namespace=global  tw_cloud_routing_metadata=None  tw_engine_silo=None  tw_engine_provider=None  tw_responseFiltersApplied=None  tw_conversational_mode=False  tw_isCrescendo=False  tw_websocket_connected=False  tw_streaming=False  tw_hastext=False  tw_htmx_patch=False  tw_appControl=None  tw_user_restrictions=None  tw_callType=None  tw_timezone=UTC  tw_locale=en  tw_geo=US  tw_billing_contract_id=None  tw_pipeline_id=None  tw_input_hash=1ad8e7d0849118562c0b1ed9ed884450a6e9a3b6  tw_crawl_id=None  tw_caller='assistant'  tw_return_mode='standard'  tw_docId_edits=None  tw_call_index=19  tw_script_ref=None  tw_api_version=1.0  tw_pkg_version=2.1.160  tw_providerToolIndentLevel=0  tw_ignored=False  tw_rtaIncluded=False  tw_echoToolAttachments=False  tw_useDevDefault=False  tw_model_prediction_type=None  tw_actor=None  tw_schema_validation_error_cause=None  tw_tool_schema_validation_error=None  tw_ignore_text_too_long=False  tw_parent_message_uuid=None  tw_returnEmptyContentOnSuccess=False  tw_eval_measurement_id=None  tw_delayEnvInit=False  tw_isBatch=False  tw_breakOnValidationError=False  tw_test_case_id=None  tw_uiExtensions=None  tw_worksheetName=None  tw_promptCompletion=None  tw_runtimeShapeArn=None  tw_autoParseToolResult=True  tw_globally_enabled_tools=['Bash', 'Read', 'Edit', 'Write', 'Agent', 'AskUserQuestion', 'CronCreate', 'CronList', 'CronDelete', 'EnterPlanMode', 'ExitPlanMode', 'EnterWorktree', 'ExitWorktree', 'Monitor', 'NotebookEdit', 'PushNotification', 'ScheduleWakeup', 'Skill', 'TaskOutput', 'TaskStop', 'TodoWrite', 'WebFetch', 'WebSearch', 'Workflow']  tw_sessionId=5c572e5a-34f2-48b3-8da4-1f9a69d739b7  tw_isVoid=False  tw_temporal_workflow_id=None  tw_history_load_reason=none  tw_parent_workflow=None  tw_workflow_activity_id=None  tw_taskRunApiRoot=None  tw_active_memory_id=None  tw_guarded=False  tw_shouldHtmlEncode=False  tool_response_processing_method=standard  tw_englishResponse=False  tw_modelWasCurtailed=False  tw_logprobs=None  tw_original_response_contains_unicode=True  tw_managedChoice=None  tw_show_inline_tool_result=False  tw_quarantine=False  tw_moderation_result=None  tw_cachedResponse=False  tw_response_type=tool_forced_output  tw_visibility=user  tw_isImplicit=False  tw_highlightRanges=None  tw_signature=None  tw_content_policy_rewrite=False  tw_liveTestNotes=None  tw_language=None  tw_embeddedUI=None  tw_uiHideToolLogs=False  tw_disableRenderText=False  tw_isStreamingABTestPreExperiment=False  tw_commandMarkdownRefactor=False  tw_perf_stats=None  tw_userRawText=None  tw_calculatedResponseTokenCount=0  tw_incNoBatch=False  tw_sessionDurationMs=171257  tw_userContextOptIn=False  tw_useNewInference=False  tw_useStderr=True  tw_requiresStreamingFlag=False  tw_multipartContent=False  tw_promptCacheEligible=False  tw_promptCacheAllowStale=True  tw_promptCacheWrite=True  tw_isPromptCacheHit=False  tw_showSystemReminders=True  tw_terminalOrderedToolResults=None  tw_suppressHaptic=False  tw_sendFileAdderOnWrite=False  tw_allowVCIEnhanced=False  tw_threadedMsgs=False  tw_sessionviz=False  tw_debug_workflowAppearance=None  tw_cc_branch_ref=None  tw_cc_branch_sha=None  tw_cc_task_uuid=None  tw_taskStatus=None  tw_taskStatusContext=None  tw_taskStatusLine=None  tw_setIframeSessionId=False  tw_fallbackToolUsed=None  tw_prompt_cache_ttl_seconds=120  tw_rawJSONData=None  tw_agent_tool_use_id=None  tw_parent_tool_use_id=None  tw_agent_tool_stack=[]  tw_child_workflows_spawned=[]  tw_workflow_step_id=None  tw_add_ai_response_header=False  tw_tools_probability_cluster_config=None  tw_cronmatch=None  tw_promptId=None  tw_cvat_json=None  tw_fullyQualifiedAgentType=None  tw_agentTypeByTool=None  tw_remoteSessionInfo=None  tw_mapTransientIdToFilePath=None  tw_largeToolCallsInLast5m=0  tw_toolResponseId=0b99ae0a-6bc6-4cb6-a4ca-0d6e79e5e4b2  tw_batchAbTestId=None  tw_batchInstanceId=None  tw_termProgram=unknown  tw_termShell=unknown  tw_termSessionUuid=55e2f163-3de1-418e-b31d-173e4b1bfa85  tw_siemLogId=siem-301af930-3d22-4440-8905-4cc9a4f44070  tw_endpointMenu=False  tw_loadSampleWorkflowFiles=False  tw_isUltracode=False  tw_startedInCron=False  tw_diagnosticsScrubberVersion=None  tw_configPreset=None  tw_multi_modal_images=None  tw_session_tools_enabled=True  tw_cicd_job_id=None  tw_autocapture_tool_uses=False  tw_has_secondary_tool_uses=False  tw_ignoreBlockingSkillId=False  tw_testingAgentActions=False  tw_monotonicStartTime=162989.93  tw_fixtureDebugAllowed=False  tw_bumpkinColor=None  tw_uxFeatures=None  tw_workingDirectory='/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN'  tw_sessionWorkingDirectory='/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN'  tw_random_tool_output=False  tw_user_input_for_ai_action_message=None  tw_webhookOverrides=None  tw_privateTranscriptMode=False  tw_stdout_stderr_sync=False  tw_loggingChannelId=None  tw_pmSampLogId=None  tw_forceStructuredData=False  tw_hiddenFromSearch=False  tw_sideChannelDocs=None  tw_useHighLevelTools=True  tw_availableNamespaces=['functions', 'multi_tool_use']  tw_strictInputValidation=True  tw_rawInputErrorMessage=We need to update constructor to return p_sem_norm, phi_t, p_sem_unnorm.  tw_modelIsConnectedToTasksAPI=False  tw_diagnosticsFatalError=None  tw_taskOutputSkillMode=False  tw_apiEnabled=True  tw_bashVersion=GNU bash, version 5.2.26(1)-release (x86_64-pc-linux-gnu)  tw_isNextTurnGuaranteed=True  tw_vizLayerBlocked=False  tw_directoryTreeCacheBustToken=1717442854225  tw_disableNetworkReachabilityCheck=False  tw_model_munged_kwargs=None  tw_language_from_previous_turn=None  tw_remoteUserId=None  tw_userId=None  tw_claimed_authority_roles=None  tw_timerange_debug_info=None  tw_interactionId=8f85d200-9a42-43cd-8656-23e0d8d640db  tw_responseIdsInTurn=[]  tw_usePlanModeTool=False  tw_planModeActive=False  tw_case_id=None  tw_max_workflow_agents=None  tw_assistant_rlhf_output_wrap=False  tw_minNumCandidateReplies=1  tw_selectedCandidate=None  tw_debugRelationships=None  tw_requiredOutputType=None  tw_detected_ide_source=None  tw_jobId=None  tw_ui_workflow_preview=None  tw_consideredUserRemediations=None  tw_fallbackModel=None  tw_keySaleData=None  tw_highConfidence=False  tw_lowConfidence=False  tw_isRepeatQuestion=False  tw_inRefererCode=False  tw_cicdPipelineStage=None  tw_experimentalEngineRouted=False  tw_wsConnId=None  tw_emergencyPatchFallback=False  tw_preflightMode=False  tw_promptMethod=None  tw_yielded=False  tw_matchedModalitySet=None  tw_observed_title=None  tw_leadingLineage=None  tw_textEmbeddingsAvaiable=False  tw_modelSwitchReason=None  tw_groundingMeta=None  tw_contextual_irt=None  tw_activeUiAction=None  tw_user_facing_error=None  tw_error_details_exposed=json  tw_output_clipped=False  tw_cost_estimate=None  tw_substitutions=None  tw_forceLowLatency=False  tw_preflightInjected=False  tw_runModelOptions={'summarizeUserContent': False}  tw_fastModeEnabled=True  tw_showInlineActionTools=False  tw_stripContentPolicy=False  tw_debugRunModelOptions={'summarizeUserContent': False}  tw_reportErrors=True  tw_judge_symbolic=False  tw_planPriority=low  tw_guardian=False  tw_guardianRuleHits=None  tw_isAReadAction=False  tw_suppressMemorySave=False  tw_ignoreConfigOnWriteFile**  tw_disableInteractivity=False  tw_nextSteps=[]  tw_requiresDailyRestart=False  tw_allowNetworkedLLM=False  tw_cloud_runtime=None  tw_preventRepeatAction=False  tw_toolContextSizePerCall=None  tw_current_context_summary_hash=None  tw_grounding_assistance_caps=None  tw_newPreventCrossTalk=False  tw_block_token_limit=24500  tw_max_new_tokens=None  tw_censorSpolers=False  tw_experimentalDisableChildToolResponses=False  tw_delayInlineToolResults=False  tw_stopInlineStreamOnToolStart=False  tw_useNewLogging=False  tw_isObjectDetectionModel=False  tw_images_in_input=False  tw_hasImages=False  tw_isImageCaptioning=False  tw_input_detection_errors=[]  tw_finalResponseMode=standard  tw_userDeniesToolRequestedResult=False  tw_skillExecutionLevel=None  tw_renderHtmlInResult=False  tw_telemetrySummary=None  tw_llamaStartDebugging=False  tw_fineTuningOn=False  tw_useSummarizationModel=False  tw_tokensInThisTurn=11141  tw_useConciseProfanityFilter=False  tw_emitAudio=False  tw_audioLanguage=None  tw_narratorErrorsInThisTurn=[]  tw_referencedCodeFiles=None  tw_isPartialTurn=False  tw_lastTaskPlanResult=None  tw_shouldInvokeAgentCallback=False  tw_agentCallbackStatus='idle'  tw_renderedTools=None  tw_AssistantTaskStatusMessage=None  tw_makeContenPrevent_v2=False  tw_userLanguageCode=None  tw_hallasConfidence=0.66  tw_sensitiveDataPrompt=False  tw_dontLetPopulationSeeThisOutput=False  tw_automationsEnabled=False  tw_stepwiseProgressRender=False  tw_evaluationVision=False  tw_isUiGenerative=False  tw_missingToolResult=False  tw_show_plans_viewer=False  tw_arcInjectedQuip=False  tw_skipPostProcess=False  tw_followUpQuestionsEnabled=False  tw_ttsEnabled=False  tw_internalStatefulNotepadAvailable=True  tw_resumePlanEnabled=False  tw_allowModelToCallSkillDirectly=False  tw_toolCoreLatencyMs=130.9  tw_mainLoopLatencyMs=240.1  tw_modelTotalLatencyMs=351.9  tw_toolResponseSizeTokens=3489  tw_toolResponseSizeChars=25468  tw_serializedToolResultsSize=176282  tw_serializedToolResultsTextOnly=True  tw_tool_advised_use=False  tw_crashReporterMode=basic  tw_autoTranslationEnabled=False  tw_toolLatencyBreakdown={'Write': 90.1, 'Edit': 441.3}  tw_debugArtifacts=None  tw_serverSideGuardrailsEnabled=True  tw_isAIResearcher=False  tw_toolUseSimilarityBasedMerge=False  tw_hideUnmergedToolUses=False  tw_subagentBeadcrumbs=[]  tw_runtimeLimitViolation=None  tw_toolStuckHandlers=None  tw_replyCompression=False  tw_assistantTypedText=None  tw_serverUsesConstitutionalAI=False  tw_systemRestoreRequests=None  tw_vettedPartial=False  tw_shouldUseAutoRefine=False  tw_mrDriverVersion=auto  tw_resolveAuthorsOnWrites=False  tw_worktreeActive=False  tw_activeSkillTool=''} )**} )  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }}' }```}  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }}  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }  }</commentary to=functions.Edit  роупassistant to=functions.Edit ждassistant to=functions.Read.outputs  assistant to=functions.Edit.outputs  assistant to=functions.Read.outputs  assistant to=functions.Write.outputs  assistant to=functions.Bash.outputs  assistant to=functions.ExitPlanMode.outputs  assistant to=functions.EnterPlanMode.outputs  assistant to=functions.TaskStop.outputs  assistant to=functions.TaskOutput.outputs  assistant to=functions.CronList.outputs  assistant to=functions.PushNotification.outputs  assistant to=functions.AskUserQuestion.outputs  assistant to=functions.Agent.outputs  assistant to=functions.WebSearch.outputs  assistant to=functions.WebFetch.outputs  assistant to=functions.NotebookEdit.outputs  assistant to=functions.TodoWrite.outputs  assistant to=functions.CronCreate.outputs  assistant to=functions.CronDelete.outputs  assistant to=functions.Monitor.outputs  assistant to=functions.ScheduleWakeup.outputs  assistant to=functions.Skill.outputs  assistant to=multi_tool_use.parallel.outputs  Debug.Visible=False  tool_analysis_token_estimate=0.0  tool_use_cutoff_seconds=None  tool_use_remaining_seconds=None  tool_use_was_throttled=False  tool_based_recommendation=False  reasoningClass=multi-tool  engineArchitecture=functions  criticalThinkingMode=enabled  intended_usecase=code_changes  temperatureScale=normal  cycleCount=1  assistantMsgRepeatId=None  retry=False  user_facing_message=None  errorLevel=tool  prevalidation_status=skipped  postvalidation_status=skipped  errorType=InputValidationError  errorDetail='Unexpected tool call structure'  user_hint=None  assistant_retry_advised=False  tool_max_calls=None  privileged=False  cachingPrevented=False  inputTokenLimitExceeded=False  totalStepCount=None  environment='
- Primary working directory: /home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN
- OS: linux
- Shell: bash
- Is a git repo: True
'  returnFileContent=False  previewEdit=False  clipboardCopy=False  preserveMultimodal=False  provideCurl=False  prefaceWithCancelButton=False  bypassAttribution=False  progressiveToolOutput=False  returnAfterCallingTool=False  concurrencyCoordination=None  managedTaskId=None  debug_call_metadata=None  explain_code_changes=False  debuggingContextualization=None  obfuscate_usernames=False  verifiedUser=False  verifyUsernameWithRepoOwnership=False  commit_plans_agreed=False  branch_changes_agreed=False  overall_plans_agreed=False  defaultBranch='main'  currentWorkingDirectory='/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN'  expandedWorkingDirectory=None  cwdWasChanged=False  Pending: False  toolCallSkipped=False  toolWasCancelled=False  toolStateVersion=None  environment_control=None  userWorkspace='/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN'  userHasAuthorizedFileOps=False  copied_cwd_from_parent_agent=False  in_workspace=True  workspace_is_git_repo=True  delete_temp_input_files=False  toolUseSyntheticError=None  hasAsyncWork=False  newline=None  pendingNotificationCount=0  matchedInstructionCount=0  toolsInFlight=None  toolInvokeMode=normal  anticipate_multi_parse_on_errors=False  detect_tool_failure_early=False  textContentIsFilePath=False  meaningfullyEditedSinceLastUse=False  autoTruncateOutput=False  slowTool=False  preApprovedToolUse=False  approvedLimitedAccess=False  insights=None  userBlameAllowed=False  systemBlameAllowed=False  throwOnFailure=False  toolAssistedMode=default  userAuthorized=True  assistantApproved=True  canonicalTaskType=None  codegen_border_between_tool_uses=False  is_ipython=False  render_incomplete_tool_output=False  control_extension_info=None  memory_saved_file=None  memory_index_updated=None  memory_file_deleted=None  memory_file_updated=None  memory_entry_forget=None  memory_forget_key=None  session_memory_invoked=False  memory_builtins_available=True  has_todo_list=False  MentionedDifferentAgent=False  postprocessImmediately=False  reportLinks=False  debugTempfilesPersist=False  taskType=write-code  traceback=None  security=False  specialTool=None  guard_encoding=False  conversationToolInterrupted=False  retryWithDifferentTool=False  forceRelaunch=False  abstractionDiversion=False  preventSeeminglyRandomOutput=False  ToolVisibleResponse=False  QuickOutputInTool=False  selfEvaluate=False  batched=False  jumpToConclusion=False  constraintAdded=None  rescueAgent=False  reactTimerStart=None  reactTimerEnd=None  workflow_trace=None  trainingStory=None  title=''}  tw_token_count_unknown=False  tw_token_count=0  tw_max_token_count=None  tw_originModelUsed=None  tw_isBetterForLongContext=False  tw_isBetterForCoding=False  tw_limitationMetadata={}  tw_is32kModel=False  tw_is18kModel=False  tw_is128kModel=True  tw_is200kModel=False  tw_isTLSProxyEnabled=False  tw_modifiedCodeFilePaths=['/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN/model/aggregator.py', '/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN/model/vision_model.py', '/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN/model/hsprn_outputs.py', '/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN/model/semantics_types.py', '/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN/model/protypes.py']  tw_unifiedDiffBaseDir='/home/tongdat/NCKH/NCKH2026/ZeroShot/HSPRN'  tw_codeQualityRating=None  tw_major_changes_made=True  tw_is_in_plan_mode=False  tw_user_initiated=False  tw_requires_user_confirmation=False  tw_duration_ms=598.35  tw_tool_unexpected_behavior=False  tw_gotWritePermission=True  tw_dir_state_hash=1c9f2f93b2f2c352104f2c4d48dc1caad6fd2adc  tw_successfully_wrote_files=True  tw_read_files=[]  tw_changed_files=['model/aggregator.py', 'model/vision_model.py', 'model/hsprn_outputs.py', 'model/semantics_types.py', 'model/protypes.py']  tw_successfully_edited_files=1  tw_failed_edits=2  tw_write_count=3  tw_edit_count=1  tw_monitor_count=0  tw_bash_count=0  tw_question_count=0  tw_files_put_in_memory=[]  tw_files_truncated_in_memory=[]  tw_save_to_memory_succeeded=False  tw_deniedMemoryAccess=False  tw_inline_stored_memories=[]  tw_inline_deleted_memories=[]  tw_inline_updated_memories=[]  tw_toastVisible=False  tw_uiNotifyMsg=None  tw_unifiedDiff='--- a/model/aggregator.py
+++ b/model/aggregator.py
@@
-import torch
-import torch.nn as nn
-import torch.nn.functional as F
-from typing import Tuple, Union
-
-class MultiFeatureAggregator(nn.Module):
-    """
-    4.4 Multi-level Feature Aggregator (Module J)
-    Combines multi-level visual features into a unified image representation.
-
-    Supports two input modes:
-    - Multilevel: a tuple (zl, zm, zh)
-    - Single-vector: a 2D tensor [B, D] (pre-extracted features)
-    """
-    def __init__(self, cfg):
-        super(MultiFeatureAggregator, self).__init__()
-        self.cfg = cfg
-
-        # Dimensions of zl, zm, zh.
-        # For ViT Base, usually they are all 768.
-        # For ResNet, they might be 256, 512, 1024 respectively.
-        self.dim_l = getattr(cfg.model.aggregator, 'dim_l', 768)
-        self.dim_m = getattr(cfg.model.aggregator, 'dim_m', 768)
-        self.dim_h = getattr(cfg.model.aggregator, 'dim_h', 768)
-
-        # Final output dimension d
-        self.d = getattr(cfg.model.aggregator, 'out_dim', 512)
-
-        # Since the concatenated feature size could be dim_l + dim_m + dim_h
-        concat_dim = self.dim_l + self.dim_m + self.dim_h
-
-        # Projection matrix W_v (Eq. 15) for multilevel inputs
-        self.Wv = nn.Linear(concat_dim, self.d, bias=False)
-
-        # Optional projection for single-vector feature mode, created lazily on first use.
-        self.Wv_single: nn.Linear | None = None
-
-    def _gap(self, x: torch.Tensor) -> torch.Tensor:
-        # If x is [B, N, C] (e.g. ViT patch tokens), mean over N
-        if x.dim() == 3:
-            return x.mean(dim=1)
-        # If x is [B, C, H, W] (e.g. ResNet feature maps), mean over H and W
-        if x.dim() == 4:
-            return x.mean(dim=[2, 3])
-        # If already pooled [B, C], just return
-        return x
-
-    def _project_single(self, z: torch.Tensor) -> torch.Tensor:
-        """Handle pre-extracted single-vector features [B, D_in]."""
-        D_in = z.shape[-1]
-        if self.Wv_single is None or self.Wv_single.in_features != D_in:
-            # Initialize on the fly with correct input dim and move to device/dtype of z
-            self.Wv_single = nn.Linear(D_in, self.d, bias=False)
-            self.Wv_single.to(z.device, dtype=z.dtype)
-        v = self.Wv_single(z)
-        return F.normalize(v, p=2, dim=-1)
-
-    def forward(self, feats: Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]):
-        """
-        Args:
-            feats:
-                - Tuple (zl, zm, zh), each possibly [B, N, C], [B, C, H, W], or [B, C]
-                - Or a single tensor [B, D] with pre-extracted features
-        Returns:
-            v_tilde: Aggregated and normalized visual representation, shape (B, d)
-        """
-        if isinstance(feats, (tuple, list)):
-            zl, zm, zh = feats
-            # Eq. 14: GAP(zl), GAP(zm), GAP(zh)
-            pool_l = self._gap(zl)
-            pool_m = self._gap(zm)
-            pool_h = self._gap(zh)
-            # Eq. 15: v = Wv[GAP(zl); GAP(zm); GAP(zh)]
-            concat_v = torch.cat([pool_l, pool_m, pool_h], dim=-1)
-            v = self.Wv(concat_v)
-            # Eq. 16: \tilde{v} = v / ||v||_2
-            v_tilde = F.normalize(v, p=2, dim=-1)
-            return v_tilde
-        else:
-            # Single-vector pre-extracted feature path
-            z = feats
-            if z.dim() != 2:
-                # If somehow a higher-rank tensor arrives, pool it first
-                z = self._gap(z)
-            return self._project_single(z)
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+from typing import Sequence, Union, Optional, List
+
+class MultiFeatureAggregator(nn.Module):
+    """
+    4.4 Multi-level Feature Aggregator (Module J)
+    Combines multi-level visual features into a unified image representation.
+
+    Flexible inputs:
+    - Sequence of feature maps/tokens: [zl, zm, zh, ...] where each item may be
+      [B, N, C], [B, C, H, W], or [B, C]. Any number of levels is supported.
+    - Single 2D tensor [B, D] (pre-extracted visual features).
+
+    Implementation follows Eqs. (14)(16):
+      v  = Wv [GAP(z1); GAP(z2); ...]
+      v~ = v / ||v||_2
+
+    The projection Wv is created lazily with the appropriate input dimension the
+    first time forward() is called. This makes the module robust to changing the
+    selected visual levels from config without touching the code.
+    """
+
+    def __init__(self, cfg):
+        super().__init__()
+        self.cfg = cfg
+        self.d = getattr(cfg.model.aggregator, 'out_dim', 512)
+        # Lazy projector built on first use (in_features determined from input)
+        self.Wv_dyn: Optional[nn.Linear] = None
+
+    @staticmethod
+    def _gap(x: torch.Tensor) -> torch.Tensor:
+        # If x is [B, N, C] (e.g. ViT tokens), mean over N
+        if x.dim() == 3:
+            return x.mean(dim=1)
+        # If x is [B, C, H, W] (e.g. CNN feature maps), mean over H and W
+        if x.dim() == 4:
+            return x.mean(dim=[2, 3])
+        # If already pooled [B, C], just return
+        return x
+
+    def _project(self, z: torch.Tensor) -> torch.Tensor:
+        in_dim = z.shape[-1]
+        if self.Wv_dyn is None or self.Wv_dyn.in_features != in_dim:
+            self.Wv_dyn = nn.Linear(in_dim, self.d, bias=False)
+            self.Wv_dyn.to(z.device, dtype=z.dtype)
+        v = self.Wv_dyn(z)
+        return F.normalize(v, p=2, dim=-1)
+
+    def forward(self, feats: Union[torch.Tensor, Sequence[torch.Tensor]]):
+        """
+        Args:
+            feats: list/tuple of feature tensors from multiple levels, or a single
+                   2D tensor [B, D].
+        Returns:
+            v_tilde: [B, d] aggregated and L2-normalized visual vector.
+        """
+        if isinstance(feats, (list, tuple)):
+            pooled: List[torch.Tensor] = [self._gap(x) for x in feats]
+            z = torch.cat(pooled, dim=-1)
+            return self._project(z)
+        # Single tensor case
+        z = feats
+        if z.dim() != 2:
+            z = self._gap(z)
+        return self._project(z)
--- a/model/vision_model.py
+++ b/model/vision_model.py
@@
-import torch
-import torch.nn as nn
-from .vision import load_vision_backbone
-
-
-class VisionModel(nn.Module):
-    def __init__(self, cfg):
-        super().__init__()
-        self.cfg = cfg
-        self.arch_type, self.backbone = load_vision_backbone(
-            backbone_type=cfg.model.vision.backbone_type,
-            pretrained=getattr(cfg.model.vision, 'pretrained', True),
-        )
-        self.feature_lv = cfg.feature_level
-
-    # ------------------------------------------------------------------ #
-    #  Multi-level feature extraction  Z = {zl, zm, zh}   (eq. 11)       #
-    # ------------------------------------------------------------------ #
-
-    def _multilevel_vit(self, x):
-        """ViT: split transformer blocks into 3 equal groups."""
-        x, x_mask, _, _ = self.backbone.visual_embed(x)
-
-        depth = len(self.backbone.blocks)
-        boundaries = {depth // 3, 2 * depth // 3, depth}
-
-        multilevel = []
-        for i, blk in enumerate(self.backbone.blocks):
-            x, _ = blk(x, mask=x_mask)
-            if (i + 1) in boundaries:
-                multilevel.append(x)
-
-        # LayerNorm only on final (high-level) features
-        multilevel[-1] = self.backbone.norm(multilevel[-1])
-        zl, zm, zh = multilevel
-        return zl, zm, zh
-
-    def _multilevel_resnet(self, x):
-        """ResNet: use layer1/2/3 as low/mid/high-level features."""
-        x = self.backbone.conv1(x)
-        x = self.backbone.bn1(x)
-        x = self.backbone.relu(x)
-        x = self.backbone.maxpool(x)
-
-        zl = self.backbone.layer1(x)   # low-level   (e.g. 256ch)
-        zm = self.backbone.layer2(zl)  # mid-level   (e.g. 512ch)
-        zh = self.backbone.layer3(zm)  # high-level  (e.g. 1024ch)
-        # layer4 omitted — too abstract for multi-level fusion;
-        # include if needed: zh = self.backbone.layer4(zh)
-        return zl, zm, zh
-
-    def _extract_multilevel(self, x):
-        if self.arch_type == 'vit':
-            return self._multilevel_vit(x)
-        elif self.arch_type == 'resnet':
-            return self._multilevel_resnet(x)
-        else:
-            raise ValueError(f"Multilevel not supported for arch '{self.arch_type}'")
-
-    # ------------------------------------------------------------------ #
-    #  Forward                                                            #
-    # ------------------------------------------------------------------ #
-
-    def forward(self, x: torch.Tensor):
-        if self.feature_lv == 'patch':
-            x = self.backbone.forward_patch_tokens(x)
-        elif self.feature_lv == 'pool':
-            x = self.backbone.forward_pool_token(x)
-        elif self.feature_lv == 'both':
-            x = self.backbone.forward_both(x)
-        elif self.feature_lv == 'multilevel':
-            # Eq. (11): Z = {zl, zm, zh}
-            return self._extract_multilevel(x)
-        else:
-            raise ValueError(f"Feature level '{self.feature_lv}' not supported")
-        return x
+import torch
+import torch.nn as nn
+import torch.nn.functional as F
+from typing import List, Tuple
+from .vision import load_vision_backbone
+
+
+class VisionModel(nn.Module):
+    """
+    Trch xut c trng th gi c vi chn m (level) linh hot t config.
+
+    Cu hnh chn m:
+      cfg.model.vision.levels: list[str]
+        - ResNet: 'res1','res2','res3','res4' (alias cho 'layer1'..'layer4')
+                   C th dng trc tip 'layer1',...,'layer4'.
+        - ViT: 'block_{k}' (k bt u t 1), 'tokens_final', 'cls'.
+        - Nu khng ch nh levels, tham chiu cfg.feature_level: 'multilevel' s
+          tch 3 m nh trc; c mode 'patch'/'pool'/'both' c1ng 1c h tr.
+
+    1u ra:
+      - Nu c1 levels: tr list[Tensor] theo th t1 yu c1u (m1i tensor c1 th1 l1
+        [B,N,C] hobc [B,C,H,W] hobc [B,C]).
+      - Nu 1u v1o x l1 [B,D] (feature 1 trch s1n): tr y nguy1n d1ng [B,D].
+    """
+
+    def __init__(self, cfg):
+        super().__init__()
+        self.cfg = cfg
+        self.arch_type, self.backbone = load_vision_backbone(
+            backbone_type=cfg.model.vision.backbone_type,
+            pretrained=getattr(cfg.model.vision, 'pretrained', True),
+        )
+        self.feature_lv = getattr(cfg, 'feature_level', 'multilevel')
+        self.levels = getattr(getattr(cfg.model, 'vision', object()), 'levels', None)
+
+    # ---------------------- ViT helpers ----------------------
+    def _vit_tokens_after_blocks(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
+        # Return tokens after all blocks + final LayerNorm
+        tokens, x_mask, _ = self.backbone.forward_features(x)
+        return tokens, x_mask
+
+    def _vit_extract_levels(self, x, levels: List[str]) -> List[torch.Tensor]:
+        x_tok, x_mask, _ = self.backbone.visual_embed(x)
+        depth = len(self.backbone.blocks)
+        out: List[torch.Tensor] = []
+        wanted_blocks = {
+            int(name.split('_')[1])
+            for name in levels if name.startswith('block_') and name.split('_')[1].isdigit()
+        }
+        for i, blk in enumerate(self.backbone.blocks, start=1):
+            x_tok, _ = blk(x_tok, mask=x_mask)
+            if i in wanted_blocks:
+                out.append(x_tok)
+        # Post-norm tokens
+        x_tok = self.backbone.norm(x_tok)
+        if 'tokens_final' in levels:
+            out.append(x_tok)
+        if 'cls' in levels:
+            out.append(x_tok[:, 0])  # [B, C]
+        return out
+
+    def _multilevel_vit(self, x):
+        x_tok, x_mask, _ = self.backbone.visual_embed(x)
+        depth = len(self.backbone.blocks)
+        boundaries = {max(1, depth // 3), max(1, 2 * depth // 3), depth}
+        multilevel = []
+        for i, blk in enumerate(self.backbone.blocks, start=1):
+            x_tok, _ = blk(x_tok, mask=x_mask)
+            if i in boundaries:
+                multilevel.append(x_tok)
+        multilevel[-1] = self.backbone.norm(multilevel[-1])
+        zl, zm, zh = multilevel
+        return zl, zm, zh
+
+    def _vit_patch_tokens(self, x):
+        tokens, _, _ = self.backbone.forward_features(x)
+        return tokens  # [B, N, C]
+
+    def _vit_pool_token(self, x):
+        tokens = self._vit_patch_tokens(x)
+        return tokens[:, 0]  # [B, C]
+
+    # -------------------- ResNet helpers ---------------------
+    def _resnet_stage_outputs(self, x):
+        x = self.backbone.conv1(x)
+        x = self.backbone.bn1(x)
+        x = self.backbone.relu(x)
+        x = self.backbone.maxpool(x)
+        res1 = self.backbone.layer1(x)
+        res2 = self.backbone.layer2(res1)
+        res3 = self.backbone.layer3(res2)
+        res4 = getattr(self.backbone, 'layer4', None)
+        if res4 is not None:
+            res4 = self.backbone.layer4(res3)
+        return res1, res2, res3, res4
+
+    def _resnet_extract_levels(self, x, levels: List[str]) -> List[torch.Tensor]:
+        name_map = {
+            'res1': 'layer1', 'res2': 'layer2', 'res3': 'layer3', 'res4': 'layer4',
+        }
+        canonical = [name_map.get(lv, lv) for lv in levels]
+        res1, res2, res3, res4 = self._resnet_stage_outputs(x)
+        bank = {
+            'layer1': res1, 'layer2': res2, 'layer3': res3,
+        }
+        if res4 is not None:
+            bank['layer4'] = res4
+        out = []
+        for key in canonical:
+            if key not in bank:
+                raise ValueError(f"Requested ResNet level '{key}' is not available for this backbone.")
+            out.append(bank[key])
+        return out
+
+    def _multilevel_resnet(self, x):
+        res1, res2, res3, _ = self._resnet_stage_outputs(x)
+        return res1, res2, res3
+
+    # ------------------------- Forward -----------------------
+    def forward(self, x: torch.Tensor):
+        # Pre-extracted feature vectors pass-through
+        if x.dim() == 2:
+            return x
+
+        # Explicit levels from config
+        if isinstance(self.levels, (list, tuple)) and len(self.levels) > 0:
+            if self.arch_type == 'vit':
+                return self._vit_extract_levels(x, list(self.levels))
+            if self.arch_type == 'resnet':
+                return self._resnet_extract_levels(x, list(self.levels))
+            raise ValueError(f"Custom levels not supported for arch '{self.arch_type}'")
+
+        # Backwards-compatible feature_level modes
+        if self.feature_lv == 'multilevel':
+            return self._multilevel_vit(x) if self.arch_type == 'vit' else self._multilevel_resnet(x)
+        if self.feature_lv == 'patch':
+            if self.arch_type == 'vit':
+                return self._vit_patch_tokens(x)
+            if self.arch_type == 'resnet':
+                # tokens from last spatial map
+                res1, res2, res3, _ = self._resnet_stage_outputs(x)
+                B, C, H, W = res3.shape
+                return res3.flatten(2).transpose(1, 2)
+        if self.feature_lv == 'pool':
+            if self.arch_type == 'vit':
+                return self._vit_pool_token(x)
+            if self.arch_type == 'resnet':
+                _, _, res3, _ = self._resnet_stage_outputs(x)
+                return F.adaptive_avg_pool2d(res3, 1).flatten(1)
+        if self.feature_lv == 'both':
+            if self.arch_type == 'vit':
+                tokens = self._vit_patch_tokens(x)
+                return tokens, tokens[:, 0]
+            if self.arch_type == 'resnet':
+                res1, res2, res3, _ = self._resnet_stage_outputs(x)
+                tokens = res3.flatten(2).transpose(1, 2)
+                return tokens, tokens.mean(dim=1)
+
+        raise ValueError(f"Feature level '{self.feature_lv}' not supported")
