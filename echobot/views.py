from django.conf import settings
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextSendMessage, TextMessage

from .vividbot import VividBot

line_bot_api = LineBotApi(settings.LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(settings.LINE_CHANNEL_SECRET)

vividbot = VividBot()


@handler.add(MessageEvent, message=TextMessage)
def _handle_message(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=vividbot.feed(event.message.text))
    )


@handler.default()
def _default(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text='Currently supports only text message')
    )


@csrf_exempt
@require_POST
def callback(request):
    """
    Callback function for Line
    """
    signature = request.META['HTTP_X_LINE_SIGNATURE']
    body = request.body.decode('utf-8')

    try:
        # This handler use _handle_message and _default to handle message from
        # line server
        handler.handle(body, signature)
    except InvalidSignatureError:
        return HttpResponseForbidden()
    except LineBotApiError:
        return HttpResponseBadRequest()

    return HttpResponse()


@csrf_exempt
@require_POST
def direct_callback(request):
    """
    Callback function for direct post
    """
    text = request.POST['text']
    bot_response = vividbot.feed(text)
    return JsonResponse({'text': bot_response})
