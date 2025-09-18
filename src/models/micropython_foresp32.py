# main.py — MicroPython ESP32: Narada MQTT -> USB-Serial (one-way), silent after connect
import sys, time, network, machine
from umqtt.simple import MQTTClient

# -------- Wi-Fi --------
WIFI_SSID = "S20plus"
WIFI_PASS = "12345789"

# -------- MQTT (Narada) --------
MQTT_HOST = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_USER = None
MQTT_PASS = None

TOPIC_IN = b"sentiment/in"   # الهاتف ينشر هنا (سطر نص واحد)

KEEPALIVE_S = 60

def wifi_connect():
    sta = network.WLAN(network.STA_IF)
    if not sta.active():
        sta.active(True)
    if not sta.isconnected():
        sta.connect(WIFI_SSID, WIFI_PASS)
        t0 = time.ticks_ms()
        while not sta.isconnected():
            if time.ticks_diff(time.ticks_ms(), t0) > 15000:
                raise OSError("Wi-Fi connect timeout")
            time.sleep_ms(200)
    print("Wi-Fi connected:", sta.ifconfig()[0])

def on_mqtt_msg(topic, msg):
    # تأكد أنها سطر واحد
    if b"\n" in msg or b"\r" in msg:
        msg = msg.replace(b"\r", b" ").replace(b"\n", b" ")
    # مرّرها للابتوب كسطر UTF-8 منتهي بـ \n
    try:
        sys.stdout.write(msg.decode("utf-8", "replace") + "\n")
        sys.stdout.flush()
    except Exception:
        pass  # تجاهل الحالات النادرة للترميز

def mqtt_connect_and_subscribe():
    cid = b"esp32-bridge-" + machine.unique_id()
    c = MQTTClient(client_id=cid, server=MQTT_HOST, port=MQTT_PORT,
                   user=MQTT_USER, password=MQTT_PASS,
                   keepalive=KEEPALIVE_S, ssl=False)
    c.set_callback(on_mqtt_msg)
    c.connect()
    c.subscribe(TOPIC_IN)
    return c

def run():
    # 1) Wi-Fi
    wifi_connect()

    # 2) MQTT (أول اتصال + طباعة مرة واحدة)
    client = None
    while client is None:
        try:
            client = mqtt_connect_and_subscribe()
        except Exception:
            time.sleep_ms(1000)
    print("MQTT connected:", MQTT_HOST, "subscribed to", TOPIC_IN.decode())

    last_ping = time.ticks_ms()

    # 3) حلقة التشغيل: رسائل / Ping / إعادة اتصال صامتة
    while True:
        # استقبل رسائل الهاتف
        try:
            client.check_msg()  # يستدعي on_mqtt_msg
        except Exception:
            # إعادة اتصال صامتة (لا تطبع شيئًا)
            ok = False
            while not ok:
                try:
                    client = mqtt_connect_and_subscribe()
                    ok = True
                except Exception:
                    time.sleep_ms(1000)

        # Ping دوري للحفاظ على الجلسة
        if time.ticks_diff(time.ticks_ms(), last_ping) > 30000:
            try:
                client.ping()
            except Exception:
                # سيُعاد الاتصال أعلاه بصمت
                pass
            last_ping = time.ticks_ms()

        time.sleep_ms(10)

run()

